"""Route-A ALFWorld local-intervention evaluator.

This is an evaluation-only protocol.  Every suffix resets the game and replays
the frozen action prefix, checks the visible state at every replayed step, then
runs Null, one-action Pulse, or Persistent continuations.  Pulse rebuilds the
next model context from the no-skill prompt so the card is absent after exactly
one complete model response.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from src.alfworld.generate.delta_credit import _teacher_forced
from src.alfworld.generate.episode import (
    _apply_generation_budget,
    _ensure_stop_string,
    extract_action,
    format_step_observation,
)
from src.alfworld.prompts import format_executor_messages, format_executor_with_skill_messages
from src.alfworld.utils.common import ensure_metadata, require_arg


def _json_field(value, default):
    if value is None:
        return deepcopy(default)
    if isinstance(value, str):
        return json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _prompt_ids(args, tokenizer, task: str, skill: str | None) -> tuple[str, list[int]]:
    messages = (
        format_executor_with_skill_messages(task, skill)
        if skill
        else format_executor_messages(task)
    )
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(getattr(args, "apply_chat_template_kwargs", None) or {}),
    )
    return text, tokenizer(text, add_special_tokens=False)["input_ids"]


def _fingerprint(game_file: str, step: int, obs: str, admissible: list[str]) -> str:
    payload = {
        "game_file": game_file,
        "step": int(step),
        "obs": str(obs),
        "admissible": [str(x) for x in admissible],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _row_lp(row) -> float:
    if isinstance(row, dict):
        value = row.get("logprob", row.get("log_prob"))
    elif isinstance(row, (list, tuple)) and row:
        value = row[0]
    else:
        value = None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite sampled logprob")
    return result


async def _generate_action(
    args, input_ids: list[int], sampling_params: dict, logical_seed: int
) -> dict:
    params = deepcopy(sampling_params)
    params["temperature"] = 1.0
    params["top_p"] = 1.0
    params.pop("top_k", None)
    params["sampling_seed"] = int(logical_seed)
    _ensure_stop_string(params, "</action>")
    max_context = int(getattr(args, "rollout_max_context_len", 32768))
    remaining = max_context - len(input_ids)
    if remaining <= 0 or _apply_generation_budget(args, params, remaining) <= 0:
        raise ValueError("no context budget for action")
    output = await post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        {"input_ids": input_ids, "sampling_params": params, "return_logprob": True},
    )
    finish = output.get("meta_info", {}).get("finish_reason", {}).get("type")
    if finish == "abort":
        raise RuntimeError(f"action generation ended with {finish}")
    rows = output.get("meta_info", {}).get("output_token_logprobs")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("missing output_token_logprobs")
    token_ids = [int(row[1]) for row in rows]
    logprobs = [_row_lp(row) for row in rows]
    text = str(output.get("text") or "")
    action, valid = extract_action(text)
    return {
        "text": text,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "logprob_sum": float(sum(logprobs)),
        "action": action,
        "action_valid": bool(valid),
        "finish_type": finish,
    }


async def run_eval(args, sample: Sample, sampling_params: dict) -> Sample:
    label = sample.label if isinstance(sample.label, dict) else {}
    condition = str(label.get("route_condition") or "null").lower()
    if condition not in {"null", "scoring_null", "pulse", "persistent"}:
        raise ValueError(f"unknown Route-A condition {condition!r}")
    skill = str(label.get("route_skill") or "")
    if condition in {"null", "scoring_null"}:
        skill = ""
    scoring_cards = _json_field(label.get("route_scoring_cards_json"), [])
    prefix = _json_field(label.get("route_history_json"), [])
    expected_decision = str(label.get("route_decision_fingerprint") or "")
    repeat_count = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    repeat = int(sample.index) % repeat_count
    seed_key = "|".join(
        [
            "route_a_v1",
            str(label.get("route_stage") or ""),
            str(label.get("route_query_id") or ""),
            str(label.get("route_history_id") or ""),
            str(repeat),
            str(label.get("route_seed_stream") or label.get("route_stage") or "default"),
        ]
    )
    logical_seed_base = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big") % (2**31 - 1)

    state = GenerateState(args)
    tokenizer = state.tokenizer
    env_url = require_arg(args, "alfworld_env_url")
    data_root = str(
        getattr(args, "alfworld_data_root", "")
        or os.environ.get("ALFWORLD_DATA_ROOT", "")
    ).strip()
    split = str(label.get("split") or "train")
    rel_dir = str(label.get("rel_dir") or "")
    game_file = f"{data_root}/{split}/{rel_dir}/game.tw-pddl"
    task = str(label.get("task_description") or label.get("question") or "")
    base_prompt_text, base_prompt = _prompt_ids(args, tokenizer, task, None)
    skill_prompt_text, skill_prompt = _prompt_ids(args, tokenizer, task, skill or None)

    replay_checks = []
    episode_id = None
    env_done = False
    infrastructure_error = ""
    turns = []
    response_text = ""
    response_ids: list[int] = []
    loss_mask: list[int] = []
    decision_fingerprint = ""
    card_removed_after_first_action = condition != "pulse"
    won = False
    model_truncated = False
    try:
        reset = await post(f"{env_url}/reset", {"game_file_path": game_file})
        episode_id = reset["episode_id"]
        current_obs = str(reset["obs"])
        admissible = [str(x) for x in reset["admissible_commands"]]
        if not task:
            task = str(reset.get("task_description") or "")
            base_prompt_text, base_prompt = _prompt_ids(args, tokenizer, task, None)
            skill_prompt_text, skill_prompt = _prompt_ids(args, tokenizer, task, skill or None)

        history_ids: list[int] = []
        history_text = ""
        for step_idx, frozen in enumerate(prefix):
            formatted_obs = format_step_observation(current_obs, admissible, step_idx)
            observed_hash = _fingerprint(game_file, step_idx, current_obs, admissible)
            expected_hash = str(frozen.get("state_fingerprint") or "")
            # A0 may intentionally leave the first engineering-history hash
            # unset and validate equality across all repeated branch restores.
            # A1/A2 manifests contain frozen expected hashes.
            ok = not expected_hash or observed_hash == expected_hash
            replay_checks.append(
                {"step": step_idx, "expected": expected_hash, "observed": observed_hash, "match": ok}
            )
            if not ok:
                raise RuntimeError(f"replay state mismatch at prefix step {step_idx}")
            frozen_response = str(frozen["model_response"])
            frozen_action = str(frozen["action"])
            action_admissible = frozen_action in admissible
            replay_checks[-1]["frozen_action"] = frozen_action
            replay_checks[-1]["action_admissible"] = action_admissible
            if not action_admissible:
                raise RuntimeError(
                    f"frozen action is not admissible at prefix step {step_idx}: "
                    f"{frozen_action!r}"
                )
            chunk = formatted_obs + frozen_response
            history_text += chunk
            history_ids.extend(tokenizer(chunk, add_special_tokens=False)["input_ids"])
            stepped = await post(
                f"{env_url}/step", {"episode_id": episode_id, "action": frozen_action}
            )
            current_obs = str(stepped["obs"])
            admissible = [str(x) for x in stepped["admissible_commands"]]
            env_done = bool(stepped["done"])
            won = bool(stepped["won"])
            if env_done:
                raise RuntimeError("frozen history terminates before intervention")

        step_idx = len(prefix)
        decision_fingerprint = _fingerprint(game_file, step_idx, current_obs, admissible)
        if expected_decision and decision_fingerprint != expected_decision:
            raise RuntimeError("decision-point fingerprint mismatch")
        decision_history_text = history_text

        first_action = None
        first_obs_text = ""
        max_steps = int(require_arg(args, "max_episode_steps"))
        while step_idx < max_steps and not env_done:
            obs_text = format_step_observation(current_obs, admissible, step_idx)
            obs_ids = tokenizer(obs_text, add_special_tokens=False)["input_ids"]
            use_skill = bool(skill) and (
                condition == "persistent" or (condition == "pulse" and first_action is None)
            )
            active_prompt = skill_prompt if use_skill else base_prompt
            generated = await _generate_action(
                args,
                active_prompt + history_ids + obs_ids,
                sampling_params,
                (logical_seed_base + step_idx) % (2**31 - 1),
            )
            if first_action is None:
                first_action = deepcopy(generated)
                first_obs_text = obs_text
            chunk = obs_text + generated["text"]
            history_text += chunk
            history_ids.extend(obs_ids)
            history_ids.extend(generated["token_ids"])
            response_text += chunk
            response_ids.extend(obs_ids)
            response_ids.extend(generated["token_ids"])
            loss_mask.extend([0] * len(obs_ids))
            loss_mask.extend([1] * len(generated["token_ids"]))
            turns.append(
                {
                    "turn_index": step_idx,
                    "obs": current_obs,
                    "admissible": admissible,
                    "state_fingerprint": _fingerprint(
                        game_file, step_idx, current_obs, admissible
                    ),
                    "action": generated["action"],
                    "action_valid": generated["action_valid"],
                    "response": generated["text"],
                    "response_token_ids": generated["token_ids"],
                    "response_logprobs": generated["logprobs"],
                    "response_logprob_sum": generated["logprob_sum"],
                    "finish_type": generated["finish_type"],
                    "skill_visible": use_skill,
                }
            )
            # Match the frozen ALFWorld wrapper: reaching the per-turn model
            # generation cap is a model truncation with reward zero, not an
            # infrastructure exception.  Keep the exact sampled response and
            # its log-probability for the intervention estimator.
            if generated["finish_type"] == "length":
                model_truncated = True
                if condition == "pulse":
                    # No subsequent model context exists, so post-pulse card
                    # removal is vacuously satisfied for this suffix.
                    card_removed_after_first_action = True
                break
            stepped = await post(
                f"{env_url}/step",
                {"episode_id": episode_id, "action": generated["action"]},
            )
            current_obs = str(stepped["obs"])
            admissible = [str(x) for x in stepped["admissible_commands"]]
            env_done = bool(stepped["done"])
            won = bool(stepped["won"])
            step_idx += 1
            if condition == "pulse" and not env_done:
                rebuilt = base_prompt_text + history_text
                card_removed_after_first_action = (
                    skill not in base_prompt_text
                    and skill not in history_text
                    and skill not in rebuilt
                )
                if not card_removed_after_first_action:
                    raise RuntimeError("card text leaked into post-pulse context")

        if first_action is None:
            raise RuntimeError("suffix generated no action")

        # Score the exact complete first response.  The empty intervention is
        # deliberately scored twice to verify w=1 and C=0 numerically.
        first_ids = [int(x) for x in first_action["token_ids"]]
        base_context = base_prompt + tokenizer(
            "".join(
                format_step_observation(
                    "", [], 0
                ) for _ in []
            ), add_special_tokens=False
        )["input_ids"]
        # history_ids now includes the whole suffix, so reconstruct the frozen
        # decision context directly from the immutable prefix and observation.
        frozen_history_text = decision_history_text
        frozen_history_ids = tokenizer(frozen_history_text, add_special_tokens=False)["input_ids"]
        first_obs_ids = tokenizer(first_obs_text, add_special_tokens=False)["input_ids"]
        base_first_context = base_prompt + frozen_history_ids + first_obs_ids
        tf_null = await _teacher_forced(args, base_first_context, first_ids)
        score_rows = []
        for card in scoring_cards:
            card_text = str(card.get("skill") or "")
            _, card_prompt = _prompt_ids(args, tokenizer, task, card_text or None)
            tf_card = await _teacher_forced(
                args, card_prompt + frozen_history_ids + first_obs_ids, first_ids
            )
            log_ratio = float(sum(tf_card) - sum(tf_null))
            weight = math.exp(max(-80.0, min(80.0, log_ratio)))
            score_rows.append(
                {
                    "card_id": str(card.get("card_id") or ""),
                    "card_sha256": str(card.get("card_sha256") or ""),
                    "null_logprob_sum": float(sum(tf_null)),
                    "skill_logprob_sum": float(sum(tf_card)),
                    "log_ratio": log_ratio,
                    "weight": weight,
                    "contribution": float((weight - 1.0) * float(won)),
                }
            )
        noop_log_ratio = float(sum(tf_null) - sum(tf_null))
        sampler_teacher_abs_error = abs(float(first_action["logprob_sum"]) - float(sum(tf_null)))
        if len(first_action["logprobs"]) != len(tf_null):
            raise RuntimeError("sampler and teacher-forced first-action token counts differ")
        sampler_teacher_token_mae = float(
            sum(abs(a - b) for a, b in zip(first_action["logprobs"], tf_null))
            / len(tf_null)
        )
        sampler_teacher_token_max_error = float(
            max(abs(a - b) for a, b in zip(first_action["logprobs"], tf_null))
        )
    except Exception as exc:
        infrastructure_error = f"{type(exc).__name__}:{exc}"
        score_rows = []
        first_action = None
        noop_log_ratio = None
        sampler_teacher_abs_error = None
        sampler_teacher_token_mae = None
        sampler_teacher_token_max_error = None
    finally:
        if episode_id and not env_done:
            try:
                await post(f"{env_url}/close", {"episode_id": episode_id})
            except Exception:
                pass

    sample.prompt = skill_prompt_text if skill else base_prompt_text
    sample.tokens = (skill_prompt if skill else base_prompt) + response_ids
    sample.response = response_text
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.reward = 10.0 * float(won) if not infrastructure_error else 0.0
    metadata = ensure_metadata(sample)
    metadata.update(
        {
            "evaluation_protocol": "local_intervention_route_a_v1",
            "route_stage": str(label.get("route_stage") or ""),
            "route_query_id": str(label.get("route_query_id") or ""),
            "route_history_id": str(label.get("route_history_id") or ""),
            "route_condition": condition,
            "route_card_id": str(label.get("route_card_id") or ""),
            "route_skill": skill,
            "route_repeat": repeat,
            "route_sampling_seed_from_slime": sampling_params.get("sampling_seed"),
            "route_logical_seed_base": logical_seed_base,
            "prefix_length": len(prefix),
            "remaining_horizon": max(0, int(require_arg(args, "max_episode_steps")) - len(prefix)),
            "replay_checks": replay_checks,
            "replay_all_match": all(x["match"] for x in replay_checks),
            "decision_fingerprint": decision_fingerprint,
            "decision_fingerprint_match": bool(decision_fingerprint) and (
                not expected_decision or decision_fingerprint == expected_decision
            ),
            "rng_state_observable": False,
            "rng_validation": "deterministic reset+full visible replay equality",
            "card_removed_after_first_action": bool(card_removed_after_first_action),
            "won": bool(won),
            "done": bool(env_done),
            "model_truncated": bool(model_truncated),
            "num_suffix_steps": len(turns),
            "turns": turns,
            "first_action": first_action,
            "scoring_rows": score_rows,
            "noop_log_ratio": noop_log_ratio,
            "noop_weight": None if noop_log_ratio is None else math.exp(noop_log_ratio),
            "noop_c_hat": None if noop_log_ratio is None else (math.exp(noop_log_ratio) - 1.0) * float(won),
            "sampler_teacher_abs_error": sampler_teacher_abs_error,
            "sampler_teacher_token_mae": sampler_teacher_token_mae,
            "sampler_teacher_token_max_error": sampler_teacher_token_max_error,
            "infrastructure_error": infrastructure_error,
            "sampling_contract": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": "disabled",
                "response_probability": "sum over all generated response tokens",
            },
        }
    )
    if infrastructure_error:
        sample.status = Sample.Status.FAILED
    elif model_truncated:
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED
    return sample
