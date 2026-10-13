"""Tau2 (dual-control) rollout generation for slime.

Used via `--custom-generate-function-path rollout.generate`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import generate as sglang_generate
from slime.utils.types import Sample

from tau2_rl_pipeline.actions import (
    QWEN3_JSON_PROTOCOL,
    QWEN35_XML_PROTOCOL,
    ToolProtocol,
    canonical_action_key,
    env_action_from_parsed_action,
    native_followup_messages_from_tau2,
    parse_assistant_action,
)
from tau2_rl_pipeline.async_env import run_tau2_env_call
from tau2_rl_pipeline.custom_tasks import base_domain_for, ensure_custom_tau2_registry
from tau2_rl_pipeline.env import (
    compute_partial_score_from_reward_info,
    max_steps_for_domain,
    parse_reward_info,
    step_with_native_assistant_message,
)
from tau2_rl_pipeline.masking import (
    QWEN_TURN_END,
    render_append_only_followup,
    render_tau2_qwen,
    strip_retained_turn_end,
)
from tau2_rl_pipeline.prompting import build_tau2_agent_system_prompt
from tau2_rl_pipeline.termination import tau2_termination_reason_from_info

TOKENIZER = None
TOOL_PROTOCOL_VALIDATED = False

# ---- Success-trajectory dump (for later rejection-sampling SFT) ----
# Enabled via TAU2_DUMP_SUCCESS_DIR. Appends one JSON line per kept trajectory.
# Per-task cap (TAU2_DUMP_MAX_PER_TASK, default 3) keeps easy tasks from
# flooding the dataset with near-duplicate episodes.
_DUMP_LOCK = threading.Lock()
_DUMP_COUNTS: dict[str, int] = defaultdict(int)
_TAU2_LOG_LOCK = threading.Lock()
_TAU2_LOG_CONFIGURED = False


def _configure_tau2_logging() -> None:
    """Keep per-turn Tau2 logs from blocking the rollout event loop/NFS."""
    global _TAU2_LOG_CONFIGURED
    if _TAU2_LOG_CONFIGURED:
        return
    with _TAU2_LOG_LOCK:
        if _TAU2_LOG_CONFIGURED:
            return
        from loguru import logger

        logger.remove()
        logger.add(
            sys.stderr,
            level=os.environ.get("TAU2_LOG_LEVEL", "ERROR").upper(),
            backtrace=False,
            diagnose=False,
            enqueue=False,
        )
        _TAU2_LOG_CONFIGURED = True


def _maybe_dump_success(sample: Sample) -> None:
    dump_dir = os.environ.get("TAU2_DUMP_SUCCESS_DIR")
    if not dump_dir:
        return
    meta = sample.metadata or {}
    if float(sample.reward or 0.0) < float(os.environ.get("TAU2_DUMP_MIN_REWARD", "1.0")):
        return
    task_id = str(meta.get("task_id", "unknown"))
    max_per_task = int(os.environ.get("TAU2_DUMP_MAX_PER_TASK", "3"))
    record = {
        "task_id": task_id,
        "domain": meta.get("domain"),
        "reward": float(sample.reward),
        "partial_score": meta.get("partial_score"),
        "tool_sequence": meta.get("tool_sequence"),
        "initial_user_message": meta.get("initial_user_message"),
        "prompt": sample.prompt,
        "response": sample.response,
    }
    with _DUMP_LOCK:
        if _DUMP_COUNTS[task_id] >= max_per_task:
            return
        _DUMP_COUNTS[task_id] += 1
        os.makedirs(dump_dir, exist_ok=True)
        # One file per process to avoid cross-actor write interleaving on NFS.
        path = os.path.join(dump_dir, f"success_pid{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tool_protocol() -> ToolProtocol:
    protocol = os.environ.get("TAU2_TOOL_PROTOCOL", QWEN35_XML_PROTOCOL)
    if protocol not in {QWEN3_JSON_PROTOCOL, QWEN35_XML_PROTOCOL}:
        raise ValueError(f"Unsupported TAU2_TOOL_PROTOCOL: {protocol!r}")
    return protocol  # type: ignore[return-value]


def _render_agent_chat(
    tokenizer,
    messages: list[dict[str, Any]],
    tools_openai: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    restore_sampled_content: bool = False,
    enable_thinking: bool = False,
    preserve_thinking: bool = False,
) -> str:
    return render_tau2_qwen(
        tokenizer,
        messages,
        tools=tools_openai,
        add_generation_prompt=add_generation_prompt,
        protocol=_tool_protocol(),
        restore_sampled_content=restore_sampled_content,
        enable_thinking=enable_thinking,
        preserve_thinking=preserve_thinking,
    )


def _agent_thinking_options(args) -> tuple[bool, bool]:
    template_kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
    enable_thinking = template_kwargs.get("enable_thinking", False)
    preserve_thinking = template_kwargs.get("preserve_thinking", False)
    if type(enable_thinking) is not bool:
        raise ValueError(
            "apply_chat_template_kwargs.enable_thinking must be a JSON boolean"
        )
    if type(preserve_thinking) is not bool:
        raise ValueError(
            "apply_chat_template_kwargs.preserve_thinking must be a JSON boolean"
        )
    if preserve_thinking and not enable_thinking:
        raise ValueError("preserve_thinking=true requires enable_thinking=true")
    if _tool_protocol() == QWEN3_JSON_PROTOCOL and (enable_thinking or preserve_thinking):
        raise ValueError("Tau2 Qwen3 JSON rollout requires enable_thinking=false")
    return enable_thinking, preserve_thinking


def _agent_enable_thinking(args) -> bool:
    return _agent_thinking_options(args)[0]


def _get_tokenizer(args):
    global TOOL_PROTOCOL_VALIDATED, TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    if not TOOL_PROTOCOL_VALIDATED:
        protocol = _tool_protocol()
        template_kwargs = dict(getattr(args, "apply_chat_template_kwargs", None) or {})
        enable_thinking, preserve_thinking = _agent_thinking_options(args)
        probe = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        probe_tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        native = TOKENIZER.apply_chat_template(
            probe,
            tools=probe_tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        rendered = _render_agent_chat(
            TOKENIZER,
            probe,
            probe_tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
        )
        if protocol == QWEN35_XML_PROTOCOL:
            if native != rendered:
                raise ValueError(
                    "Configured tokenizer does not match the prefix-stable Qwen3.5 renderer"
                )
            if "<function=example_function_name>" not in native or "<tools>" not in native:
                raise ValueError("Configured tokenizer does not expose Qwen3.5 native XML tools")
            tool_output = (
                "<tool_call>\n<function=lookup>\n<parameter=query>\n"
                "x\n</parameter>\n</function>\n</tool_call>"
            )
            first_output = (
                "reasoning\n</think>\n\n" + tool_output
                if enable_thinking
                else tool_output
            )
        else:
            if '<tool_call>\n{"name":' not in native or "<tools>" not in native:
                raise ValueError("Configured tokenizer does not expose Qwen3 Hermes JSON tools")
            first_output = (
                '<tool_call>\n{"name": "lookup", "arguments": {"query": "x"}}\n'
                "</tool_call>"
            )
        structured_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "x"},
                    },
                }
            ],
            "_sampled_content": first_output,
        }
        followup_messages = [
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "continue"},
        ]
        later_messages = [
            *probe,
            structured_assistant,
            *followup_messages,
        ]
        later = _render_agent_chat(
            TOKENIZER,
            later_messages,
            probe_tools,
            add_generation_prompt=True,
            restore_sampled_content=True,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
        )
        append_only = rendered + first_output + render_append_only_followup(
            followup_messages,
            protocol=protocol,
            enable_thinking=enable_thinking,
        )
        if later != append_only:
            raise ValueError(
                "Selected Qwen template does not match append-only Tau2 role serialization"
            )
        if protocol == QWEN35_XML_PROTOCOL:
            shared_template_path = os.environ.get("TAU2_QWEN35_CHAT_TEMPLATE")
            if not shared_template_path:
                raise ValueError(
                    "TAU2_QWEN35_CHAT_TEMPLATE is required so rollout and evaluation "
                    "use the same Qwen3.5 serialization"
                )
            original_template = TOKENIZER.chat_template
            try:
                TOKENIZER.chat_template = Path(shared_template_path).read_text(
                    encoding="utf-8"
                )
                # The shared template is used by text-side evaluation.  It can
                # prove that the initial prompt (tools, roles, and generation
                # prefix) is identical, but it cannot reconstruct arbitrary
                # sampled assistant bytes from structured tool-call history.
                # Exact sampled bytes are deliberately retained by the
                # append-only token path checked above.
                eval_render = TOKENIZER.apply_chat_template(
                    probe,
                    tools=probe_tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            finally:
                TOKENIZER.chat_template = original_template
            if eval_render != rendered:
                raise ValueError(
                    "Shared Qwen3.5 evaluation template does not match rollout rendering"
                )
        print(
            f"[tau2 rollout] validated {protocol}, append-only native "
            f"tool/user suffixes, enable_thinking={str(enable_thinking).lower()}, "
            f"and preserve_thinking={str(preserve_thinking).lower()}"
        )
        TOOL_PROTOCOL_VALIDATED = True
    return TOKENIZER


def _get_tau2_user_sim_config() -> tuple[str, dict[str, Any]]:
    user_model = os.environ.get("TAU2_USER_MODEL", "openai/Qwen/Qwen3-4B-Instruct-2507")
    user_temperature = float(os.environ.get("TAU2_USER_TEMPERATURE", "0.0"))
    user_api_base = os.environ.get("TAU2_USER_API_BASE", "http://127.0.0.1:30001/v1")

    user_llm_args: dict[str, Any] = {"temperature": user_temperature}

    if user_api_base:
        user_llm_args["api_base"] = user_api_base
        user_llm_args["api_key"] = "dummy-key-for-local-server"
        user_llm_args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    return user_model, user_llm_args


def _get_max_identical_actions() -> int:
    return int(os.environ.get("TAU2_MAX_IDENTICAL_ACTIONS", "2"))


def _get_max_episode_tokens() -> int:
    return int(os.environ.get("TAU2_MAX_EPISODE_TOKENS", "32768"))


def _native_env_followup_messages(env) -> list[dict[str, Any]]:
    """Read the structured GymAgent observation used by Tau2's native agent."""
    agent = getattr(env, "_agent", None)
    if agent is None:
        raise RuntimeError("AgentGymEnv did not expose its initialized GymAgent")
    return native_followup_messages_from_tau2(agent.observation)


async def _generate_one_action(
    args,
    *,
    sample: Sample,
    tools_openai: list[dict[str, Any]],
    sampling_params: dict[str, Any],
) -> tuple[str, Any | None, str | None, list[int], bool]:
    """Append one on-policy action to ``sample`` and parse it for Tau2."""
    stops = list(sampling_params.get("stop") or [])
    if "</tool_call>" not in stops:
        stops.append("</tool_call>")
    sampling_params = {**sampling_params, "stop": stops, "no_stop_trim": True}

    tokenizer = _get_tokenizer(args)
    previous_response_length = int(sample.response_length or 0)
    previous_text_length = len(sample.response or "")
    sample.status = Sample.Status.PENDING
    sample = await sglang_generate(args, sample, sampling_params.copy())

    new_token_count = int(sample.response_length or 0) - previous_response_length
    sampled_token_ids = sample.tokens[-new_token_count:] if new_token_count > 0 else []
    raw_text = (sample.response or "")[previous_text_length:]
    if sample.status == Sample.Status.ABORTED:
        return raw_text, None, "sglang_aborted", sampled_token_ids, False
    if raw_text and not sampled_token_ids:
        return raw_text, None, "sglang_missing_output_tokens", [], False

    parse_text, _ = strip_retained_turn_end(tokenizer, raw_text, sampled_token_ids)
    turn_end_ids = tokenizer.encode(QWEN_TURN_END, add_special_tokens=False)
    assistant_turn_closed = bool(
        turn_end_ids
        and len(sampled_token_ids) >= len(turn_end_ids)
        and sampled_token_ids[-len(turn_end_ids) :] == turn_end_ids
    )
    try:
        parsed = parse_assistant_action(
            parse_text,
            tools_openai=tools_openai,
            protocol=_tool_protocol(),
        )
        return parse_text, parsed, None, sampled_token_ids, assistant_turn_closed
    except Exception as exc:
        return parse_text, None, str(exc), sampled_token_ids, assistant_turn_closed


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """Multi-turn rollout for tau2-bench (dual-control)."""
    _configure_tau2_logging()
    from tau2.gym.gym_agent import AgentGymEnv

    metadata = sample.metadata or {}
    task_index = metadata.get("task_index", 0)
    domain = metadata["domain"]
    base_domain = metadata.get("base_domain") or base_domain_for(domain)
    task_split = metadata.get("split", "train")
    task_id = metadata["task_id"]

    ensure_custom_tau2_registry(os.environ.get("TAU2_CUSTOM_TASKS_DIR"))

    user_llm, user_llm_args = _get_tau2_user_sim_config()
    max_steps = max_steps_for_domain(base_domain)
    max_identical_actions = _get_max_identical_actions()
    max_episode_tokens = _get_max_episode_tokens()
    response_token_reserve = int(getattr(args, "rollout_max_response_len", 1024))
    tokenizer = _get_tokenizer(args)
    agent_enable_thinking, agent_preserve_thinking = _agent_thinking_options(args)

    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    env_wait_seconds = 0.0
    agent_generation_seconds = 0.0
    started_at = time.perf_counter()
    _observation, info = await run_tau2_env_call(env.reset)
    env_wait_seconds += time.perf_counter() - started_at
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]

    policy = info.get("policy", "")

    initial_messages = _native_env_followup_messages(env)
    if not initial_messages:
        raise RuntimeError("Tau2 reset returned no native agent observation")
    initial_user_message = next(
        (
            message.get("content")
            for message in initial_messages
            if message.get("role") == "user"
        ),
        None,
    )

    # GRPO w/ Static SkillBank changes only the system-prompt context. The
    # fixed bank never observes rewards and has no update/save/resume path.
    from tau2_rl_pipeline.static_skills import get_static_skill_block

    static_skill_block = get_static_skill_block(
        base_domain, str(initial_user_message or "")
    )

    data_system_prompt = metadata.get("agent_system_prompt")
    if isinstance(data_system_prompt, str) and data_system_prompt.strip():
        # A dataset-owned prompt is authoritative and must not be rewritten.
        system_prompt = data_system_prompt
        if static_skill_block.strip():
            system_prompt = (
                f"{system_prompt.rstrip()}\n\n{static_skill_block.strip()}"
            )
        system_prompt_source = (
            "metadata.agent_system_prompt+static_skillbank"
            if static_skill_block.strip()
            else "metadata.agent_system_prompt"
        )
    else:
        # Existing Tau2 pool JSONL files use `text=[{user: task}]` only as a
        # sampler placeholder. They do not contain an agent prompt, so obtain
        # the task's operational policy from the environment without exposing
        # the hidden user_scenario/evaluation criteria to the policy model.
        system_prompt = build_tau2_agent_system_prompt(
            domain=base_domain,
            policy=policy,
            tools_openai=tools_openai,
            skill_block=static_skill_block,
        )
        system_prompt_source = (
            "tau2_environment_policy+static_skillbank"
            if static_skill_block.strip()
            else "tau2_environment_policy"
        )

    evolving_skill = metadata.get("evolving_skill")
    if isinstance(evolving_skill, str) and evolving_skill.strip():
        system_prompt = (
            f"{system_prompt.rstrip()}\n\n"
            "## Retrieved reusable skill\n"
            "Use the following learned skill as guidance when it applies. "
            "The environment policy and tool schemas remain authoritative; "
            "ignore any conflicting or task-specific claims in the skill.\n\n"
            f"{evolving_skill.strip()}"
        )
        system_prompt_source = f"{system_prompt_source}+evolving_skill"

    initial_prompt = _render_agent_chat(
        tokenizer,
        [{"role": "system", "content": system_prompt}, *initial_messages],
        tools_openai,
        add_generation_prompt=True,
        enable_thinking=agent_enable_thinking,
        preserve_thinking=agent_preserve_thinking,
    )
    initial_prompt_ids = tokenizer.encode(initial_prompt, add_special_tokens=False)
    if not hasattr(sample, "append_response_tokens"):
        raise RuntimeError(
            "Tau2 append-only rollout requires upstream slime's "
            "Sample.append_response_tokens; set SLIME_DIR to the bundled Tau-compatible Slime directory"
        )
    sample.prompt = initial_prompt
    sample.tokens = list(initial_prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.status = Sample.Status.PENDING

    sampled_assistant_tokens = 0
    tool_sequence: list[str] = []
    reward = 0.0
    reward_info: dict[str, Any] = {}
    info: dict[str, Any] = {}

    terminated = False
    env_truncated = False
    termination_reason = "max_steps"
    tau2_termination_reason: str | None = None
    env_error_type: str | None = None
    env_error: str | None = None
    format_error: str | None = None
    format_total_turns = 0
    format_ok_turns = 0
    num_env_steps = 0
    last_action_key: tuple[str, str] | None = None
    identical_action_count = 0
    format_retries = int(os.environ.get("TAU2_FORMAT_RETRIES", "0"))
    if format_retries != 0:
        raise ValueError(
            "TAU2_FORMAT_RETRIES must be 0 for append-only on-policy rollout"
        )
    for _ in range(max_steps):
        if (
            max_episode_tokens > 0
            and len(sample.tokens) + response_token_reserve > max_episode_tokens
        ):
            # A prompt that cannot admit even one model turn has no policy
            # tokens and must never enter the trainer: masked empty samples
            # make aggregate entropy/KL diagnostics NaN.  Mark it removable
            # so the always-on validity filter resamples a different group.
            if sampled_assistant_tokens == 0:
                sample.remove_sample = True
                termination_reason = "prompt_context_limit"
            else:
                termination_reason = "context_limit"
            break
        started_at = time.perf_counter()
        _assistant_text, parsed, err, sampled_token_ids, assistant_turn_closed = (
            await _generate_one_action(
                args,
                sample=sample,
                tools_openai=tools_openai,
                sampling_params=sampling_params,
            )
        )
        agent_generation_seconds += time.perf_counter() - started_at
        sampled_assistant_tokens += len(sampled_token_ids)
        if err in {"sglang_aborted", "sglang_missing_output_tokens"}:
            # engine-side abort, not the model's fault: drop the sample entirely
            sample.remove_sample = True
            termination_reason = err
            break

        format_total_turns += 1
        if parsed is None:
            format_error = err
            termination_reason = "format_error"
            break
        format_ok_turns += 1

        action_key = canonical_action_key(parsed)
        if action_key == last_action_key:
            identical_action_count += 1
        else:
            last_action_key = action_key
            identical_action_count = 1
        if max_identical_actions > 0 and identical_action_count >= max_identical_actions:
            termination_reason = "repeated_action"
            break

        if parsed.name not in ("respond", "done"):
            tool_sequence.append(parsed.name)

        # Semantically invalid actions (e.g. respond with empty content) parse
        # fine but fail validation here; treat like a format failure and keep
        # the sample for the penalty.
        try:
            # Retain validation of user-facing actions, while sending the
            # structured assistant turn to Tau2 exactly as LLMAgent does.
            env_action_from_parsed_action(parsed)
        except ValueError as exc:
            format_ok_turns -= 1
            format_error = str(exc)
            termination_reason = "invalid_action"
            break
        started_at = time.perf_counter()
        try:
            _observation, reward, terminated, env_truncated, info = (
                await run_tau2_env_call(
                    step_with_native_assistant_message,
                    env,
                    content=parsed.assistant_content,
                    tool_name=None if parsed.name == "respond" else parsed.name,
                    tool_arguments=None if parsed.name == "respond" else parsed.arguments,
                )
            )
        except Exception as exc:
            # A single Tau2 episode can fail while reconstructing/evaluating
            # its environment state (for example, when a concurrently mutated
            # DB no longer matches the recorded tool result). Do not let that
            # sample cancel asyncio.gather() and terminate the entire training
            # job. Control-flow exceptions such as CancelledError are not
            # subclasses of Exception and still propagate.
            reward = 0.0
            env_error_type = type(exc).__name__
            env_error = str(exc)
            termination_reason = "tau2_env_error"
            break
        finally:
            env_wait_seconds += time.perf_counter() - started_at
        num_env_steps += 1

        if terminated:
            reward_info = parse_reward_info(info)
            tau2_termination_reason = tau2_termination_reason_from_info(info)
            # Keep the generic fallback for older Tau2 versions that do not
            # expose SimulationRun. Current Tau2 reports user_stop, agent_stop,
            # max_steps, timeout, and the individual error classes here.
            termination_reason = tau2_termination_reason or "env_terminated"
            break
        if env_truncated:
            reward_info = parse_reward_info(info)
            termination_reason = "env_truncated"
            break

        followup_messages = _native_env_followup_messages(env)
        if not followup_messages:
            raise RuntimeError(
                "Tau2 returned no structured follow-up observation for a live episode"
            )
        followup_text = render_append_only_followup(
            followup_messages,
            protocol=_tool_protocol(),
            assistant_turn_closed=assistant_turn_closed,
            enable_thinking=agent_enable_thinking,
        )
        followup_token_ids = tokenizer.encode(followup_text, add_special_tokens=False)
        if (
            max_episode_tokens > 0
            and len(sample.tokens) + len(followup_token_ids) + response_token_reserve
            > max_episode_tokens
        ):
            termination_reason = "context_limit"
            break
        sample.append_response_tokens(
            args,
            tokens=followup_token_ids,
            trainable=False,
            text=followup_text,
        )

    sample.reward = float(reward)
    if sample.remove_sample:
        sample.status = Sample.Status.ABORTED
    else:
        sample.status = Sample.Status.COMPLETED if terminated else Sample.Status.TRUNCATED

    reward_info = reward_info or parse_reward_info(info)
    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata.update(
        {
            "domain": domain,
            "base_domain": base_domain,
            "split": task_split,
            "task_id": task_id,
            "task_index": task_index,
            "user_model": user_llm,
            "reward_info": reward_info,
            "partial_score": partial_score,
            "partial_components": partial_components,
            "tool_sequence": tool_sequence,
            "terminated": terminated,
            "env_terminated": terminated,
            "env_truncated": env_truncated,
            "termination_reason": termination_reason,
            "tau2_termination_reason": tau2_termination_reason,
            "env_error_type": env_error_type,
            "env_error": env_error,
            "num_env_steps": num_env_steps,
            "tau2_env_wait_seconds": env_wait_seconds,
            "agent_generation_seconds": agent_generation_seconds,
            "identical_action_count": identical_action_count,
            "format_error": format_error,
            "format_valid": format_error is None,
            "format_total_turns": format_total_turns,
            "format_ok_turns": format_ok_turns,
            "sampled_assistant_tokens": sampled_assistant_tokens,
            "agent_tool_protocol": _tool_protocol(),
            "observation_transport": "slime_append_only",
            "action_transport": "native_structured",
            "loss_mask_mode": "slime_append_only_v1",
            "agent_enable_thinking": agent_enable_thinking,
            "agent_preserve_thinking": agent_preserve_thinking,
            "agent_system_prompt_source": system_prompt_source,
            "initial_user_message": next(
                (
                    message.get("content")
                    for message in initial_messages
                    if message.get("role") == "user"
                ),
                None,
            ),
            "agent_chat_template": (
                os.environ.get("TAU2_QWEN35_CHAT_TEMPLATE")
                if _tool_protocol() == QWEN35_XML_PROTOCOL
                else "tokenizer_config.json:chat_template"
            ),
        }
    )
    sample.metadata.update(
        {
            "episode_tokens": len(sample.tokens),
            "max_episode_tokens": max_episode_tokens,
        }
    )
    if sample.remove_sample:
        # Engine-side aborts and responses without token metadata are retried
        # by slime. They intentionally do not have complete response-aligned
        # training metadata, so do not turn a recoverable sample removal into
        # a fatal rollout-wide alignment error.
        sample.status = Sample.Status.ABORTED
        _maybe_dump_success(sample)
        return sample
    if len(sample.tokens) != len(initial_prompt_ids) + sample.response_length:
        raise RuntimeError(
            "Append-only Tau2 token invariant failed: total tokens must equal "
            "initial prompt plus response tokens"
        )
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise RuntimeError("Append-only Tau2 loss mask is not response-token aligned")
    if sum(sample.loss_mask) != sampled_assistant_tokens:
        raise RuntimeError(
            "Append-only Tau2 mask invariant failed: trainable tokens do not "
            "equal SGLang sampled tokens"
        )
    if sample.response_length == 0 and sample.rollout_log_probs is None:
        # A valid zero-token format-failure sample has no SGLang log-prob
        # entries. Represent the empty aligned sequence explicitly.
        sample.rollout_log_probs = []
    if (
        sample.rollout_log_probs is None
        or len(sample.rollout_log_probs) != sample.response_length
    ):
        raise RuntimeError("Append-only Tau2 rollout log-probs are not response-token aligned")

    _maybe_dump_success(sample)

    return sample
