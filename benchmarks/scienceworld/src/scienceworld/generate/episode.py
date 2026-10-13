"""Asynchronous multi-turn ScienceWorld episode runner for Slime."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import re

import httpx

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.scienceworld.prompts import (
    format_executor_messages,
    format_executor_with_skill_messages,
)
from src.webshop.trajectory import (
    attach_turn_outcome,
    build_turn_record,
)
from src.webshop.utils.common import require_arg


SEMAPHORE: asyncio.Semaphore | None = None


def _get_semaphore(args) -> asyncio.Semaphore:
    global SEMAPHORE
    if SEMAPHORE is None:
        SEMAPHORE = asyncio.Semaphore(
            int(require_arg(args, "scienceworld_concurrency"))
        )
    return SEMAPHORE


def _ensure_stop_string(params: dict, stop_string: str) -> None:
    current = params.get("stop")
    if current is None:
        params["stop"] = [stop_string]
    elif isinstance(current, str):
        params["stop"] = [current] if current == stop_string else [current, stop_string]
    elif stop_string not in current:
        params["stop"] = [*current, stop_string]


def extract_action(text: str) -> tuple[str, bool]:
    """Parse one strict visible-reasoning/action response."""

    reasoning = list(
        re.finditer(r"<reasoning>(.*?)</reasoning>", text or "", re.I | re.S)
    )
    actions = list(re.finditer(r"<action>(.*?)</action>", text or "", re.I | re.S))
    if len(reasoning) != 1 or len(actions) != 1:
        return "", False
    action = actions[0].group(1).strip()
    valid = bool(
        action
        and "\n" not in action
        and not re.search(r"[\u4e00-\u9fff]", text or "")
    )
    return action, valid


def format_step_observation(obs: str, action_hints: str, step_idx: int) -> str:
    return (
        f"\n\n[Step {step_idx}]\n"
        f"Observation: {obs}\n\n"
        "Current action templates and observable objects:\n"
        f"{action_hints.strip()}\n\n"
    )


def _label(sample: Sample) -> dict:
    if not isinstance(sample.label, dict):
        raise ValueError("ScienceWorld parquet row must provide a metadata label")
    required = {"task_name", "variation_idx", "split", "max_steps"}
    missing = sorted(required - set(sample.label))
    if missing:
        raise ValueError(f"ScienceWorld label missing fields: {missing}")
    return sample.label


async def run_episode(
    args,
    sample: Sample,
    sampling_params: dict | None = None,
    skill: str | None = None,
    evaluation: bool = False,
    _stale_episode_retry: int = 0,
) -> Sample:
    if getattr(args, "partial_rollout", False):
        raise ValueError("Partial rollout is not supported for ScienceWorld episodes")

    state = GenerateState(args)
    tokenizer = state.tokenizer
    gen_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env_url = require_arg(args, "scienceworld_env_url")
    label = _label(sample)
    max_turns = min(
        int(require_arg(args, "max_episode_steps")), int(label["max_steps"])
    )
    max_context_length = int(require_arg(args, "rollout_max_context_len"))
    executor_max_response_len = int(require_arg(args, "executor_max_response_len"))
    sampling_params = dict(sampling_params or {})

    response = ""
    response_token_ids: list[int] = []
    response_log_probs: list[float] = []
    loss_mask: list[int] = []
    prompt_token_ids: list[int] = []
    turn_records: list[dict] = []
    episode_id: str | None = None
    task_description = str(label.get("task_description") or "").strip()
    task_score = 0.0
    won = False
    env_done = False
    step_idx = -1
    last_action = ""
    last_model_valid = False
    last_env_valid = False
    stale_episode_error = None
    infra_error = None

    async with _get_semaphore(args):
        try:
            reset_resp = await post(
                f"{env_url}/reset",
                {
                    "task_name": str(label["task_name"]),
                    "variation_idx": int(label["variation_idx"]),
                    "split": str(label["split"]),
                    "max_steps": int(label["max_steps"]),
                },
            )
            episode_id = str(reset_resp["episode_id"])
            current_obs = str(reset_resp["obs"])
            action_hints = str(reset_resp["action_hints"])
            task_description = str(
                reset_resp.get("task_description") or task_description
            ).strip()
            if not task_description:
                raise ValueError("ScienceWorld reset returned no task description")
        except Exception as exc:
            ident = (
                f"{label.get('task_name')}:{label.get('variation_idx')}:"
                f"{label.get('split')}"
            )
            print(f"[scienceworld/run_episode] /reset error ({ident}): {exc}")
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
            sample.tokens = []
            sample.response = ""
            sample.loss_mask = []
            return sample

        messages = (
            format_executor_with_skill_messages(task_description, skill)
            if skill
            else format_executor_messages(task_description)
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(getattr(args, "apply_chat_template_kwargs", None) or {}),
        )
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        sample.prompt = prompt_text

        for step_idx in range(max_turns):
            obs_text = format_step_observation(current_obs, action_hints, step_idx)
            obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
            remaining = max_context_length - len(prompt_token_ids) - len(response_token_ids)
            if remaining <= 0 or len(obs_ids) > remaining:
                sample.status = Sample.Status.TRUNCATED
                break
            response += obs_text
            response_token_ids.extend(obs_ids)
            loss_mask.extend([0] * len(obs_ids))
            response_log_probs.extend([0.0] * len(obs_ids))

            input_ids = prompt_token_ids + response_token_ids
            remaining = max_context_length - len(input_ids)
            request_params = deepcopy(sampling_params)
            _ensure_stop_string(request_params, "</action>")
            request_params["max_new_tokens"] = min(
                remaining, executor_max_response_len
            )
            if request_params["max_new_tokens"] <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            try:
                output = await post(
                    gen_url,
                    {"input_ids": input_ids, "sampling_params": request_params, "return_logprob": True},
                )
            except Exception as exc:
                print(f"[scienceworld/run_episode] SGLang step={step_idx}: {exc}")
                infra_error = str(exc)
                sample.status = Sample.Status.FAILED
                break

            finish_type = output["meta_info"]["finish_reason"]["type"]
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            generated = str(output["text"])
            logprob_rows = output.get("meta_info", {}).get("output_token_logprobs") or []
            if not logprob_rows:
                infra_error = "missing executor output_token_logprobs"
                sample.status = Sample.Status.FAILED
                break
            generated_ids = [int(row[1]) for row in logprob_rows]
            response_log_probs.extend(float(row[0]) for row in logprob_rows)
            response_start = len(response_token_ids)
            response += generated
            response_token_ids.extend(generated_ids)
            response_end = len(response_token_ids)
            loss_mask.extend([1] * len(generated_ids))

            last_action, last_model_valid = extract_action(generated)
            turn_record = build_turn_record(
                turn_index=step_idx,
                observation=current_obs,
                admissible_actions=[action_hints],
                response_text=generated,
                action=last_action,
                model_action_valid=last_model_valid,
                response_token_start=response_start,
                response_token_end=response_end,
            )
            turn_records.append(turn_record)
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                break

            try:
                step_resp = await post(
                    f"{env_url}/step",
                    {"episode_id": episode_id, "action": last_action},
                )
                current_obs = str(step_resp["obs"])
                action_hints = str(step_resp.get("action_hints") or "")
                env_done = bool(step_resp["done"])
                won = bool(step_resp["won"])
                task_score = float(step_resp.get("task_score", 0.0) or 0.0)
                last_env_valid = bool(step_resp.get("valid_action", False))
                attach_turn_outcome(
                    turn_record,
                    environment_action_valid=last_env_valid,
                    done=env_done,
                    won=won,
                    task_score=task_score,
                )
            except Exception as exc:
                print(f"[scienceworld/run_episode] /step step={step_idx}: {exc}")
                infra_error = str(exc)
                sample.status = Sample.Status.FAILED
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code == 404
                    and "Unknown episode_id" in exc.response.text
                ):
                    stale_episode_error = exc
                break
            if env_done:
                break

        if episode_id and not env_done:
            try:
                await post(f"{env_url}/close", {"episode_id": episode_id})
            except Exception:
                pass

    if stale_episode_error is not None and _stale_episode_retry < 1:
        sample.status = Sample.Status.PENDING
        sample.prompt = ""
        sample.tokens = []
        sample.response = ""
        sample.loss_mask = []
        return await run_episode(
            args,
            sample,
            sampling_params=sampling_params,
            skill=skill,
            evaluation=evaluation,
            _stale_episode_retry=_stale_episode_retry + 1,
        )

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response = response
    sample.response_length = len(response_token_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = response_log_probs
    assert len(response_log_probs) == len(response_token_ids) == len(loss_mask)
    sample.reward = float(won)
    sample.metadata = {
        "won": won,
        "infra_error": infra_error,
        "done": env_done,
        "task_description": task_description,
        "record_id": str(label.get("record_id") or ""),
        "task_name": str(label["task_name"]),
        "variation_idx": int(label["variation_idx"]),
        "split": str(label["split"]),
        "category": str(label["task_name"]),
        "task_type": str(label["task_name"]),
        "task_score": task_score,
        "num_steps": step_idx + 1,
        "last_action": last_action,
        "last_action_valid": bool(last_model_valid and last_env_valid),
        "outcome_reward": float(won),
        "turns": turn_records,
        "trajectory_entropy": None,
        "entropy_ranked_turn_indices": [],
        "entropy_turn_count": 0,
        "entropy_token_count": 0,
        "entropy_method": "disabled_for_success_only_delta",
    }
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED if env_done else Sample.Status.TRUNCATED
    return sample
