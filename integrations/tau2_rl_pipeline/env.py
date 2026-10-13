"""Small utilities around tau2-bench `AgentGymEnv`."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


def max_steps_for_domain(domain: str) -> int:
    """Return the rollout limit, with extra room for long banking workflows."""
    default = int(os.environ.get("TAU2_MAX_STEPS", "40"))
    if domain != "banking_knowledge":
        return default
    configured = os.environ.get("TAU2_BANKING_MAX_STEPS")
    return int(configured) if configured is not None else max(default, 80)


def step_with_native_assistant_message(
    env: Any,
    *,
    content: str | None,
    tool_name: str | None,
    tool_arguments: dict[str, Any] | None,
) -> tuple[str, float, bool, bool, dict[str, Any]]:
    """Advance AgentGymEnv with the same structured message used by LLMAgent.

    AgentGymEnv.step accepts only a string and reparses it, which cannot
    represent an assistant turn containing both leading content and a tool
    call. Native Tau2 evaluation receives that structured turn directly from
    LiteLLM. Use AgentGymEnv's synchronization path without its lossy string
    conversion so the recorded/evaluated trajectory matches native eval.
    """
    from tau2.data_model.message import AssistantMessage, ToolCall

    tool_calls = None
    if tool_name is not None:
        tool_calls = [
            ToolCall(
                id="",
                name=tool_name,
                arguments=tool_arguments or {},
                requestor="assistant",
            )
        ]
    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        raw_data={"source": "tau2_rl_native_rollout"},
    )

    with env._lock:
        if env._simulation_done.is_set():
            return "", 0.0, True, False, env._get_info()

        env._agent.set_action(message)
        timeout_seconds = float(
            os.environ.get("TAU2_ENV_STEP_TIMEOUT_SECONDS", "0")
        )
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        )
        while not env._simulation_done.is_set() and not env._agent.is_agent_turn:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Tau2 environment did not return control within "
                        f"{timeout_seconds:.1f}s"
                    )
                env._simulation_done.wait(timeout=min(0.01, remaining))
            else:
                env._simulation_done.wait(timeout=0.01)

        terminated = env._simulation_done.is_set()
        observation = env._format_observation(env._agent.observation)
        reward, reward_info = env._get_reward()
        info = env._get_info()
        info["reward_info"] = reward_info
        return observation, reward, terminated, False, info


@dataclass(frozen=True, slots=True)
class Tau2EpisodeConfig:
    domain: str
    task_split: str
    user_llm: str
    user_llm_args: dict[str, Any]
    max_steps: int = 40
    all_messages_as_observation: bool = False


class Tau2TaskIndex:
    def __init__(self, domain: str, task_split: str):
        from tau2.registry import registry

        self.domain = domain
        self.task_split = task_split
        self._tasks = registry.get_tasks_loader(domain)(task_split)
        if not self._tasks:
            raise ValueError(f"No tasks loaded for domain={domain} split={task_split}")

    def __len__(self) -> int:
        return len(self._tasks)

    def task_id(self, task_index: int) -> str:
        if task_index < 0 or task_index >= len(self._tasks):
            raise IndexError(f"task_index={task_index} out of range (0..{len(self._tasks)-1})")
        return self._tasks[task_index].id


class Tau2AgentGymWrapper:
    def __init__(self, cfg: Tau2EpisodeConfig):
        self.cfg = cfg
        self._task_index = Tau2TaskIndex(cfg.domain, cfg.task_split)
        self._env = None
        self._task_id: str | None = None
        self._tools_openai: list[dict[str, Any]] | None = None
        self._policy: str | None = None

    @property
    def task_count(self) -> int:
        return len(self._task_index)

    def task_id_from_index(self, task_index: int) -> str:
        return self._task_index.task_id(task_index)

    @property
    def tools_openai_schema(self) -> list[dict[str, Any]]:
        if self._tools_openai is None:
            raise RuntimeError("Call reset() first to populate tool schemas.")
        return self._tools_openai

    @property
    def policy(self) -> str:
        if self._policy is None:
            raise RuntimeError("Call reset() first to populate policy.")
        return self._policy

    @property
    def task_id(self) -> str:
        if self._task_id is None:
            raise RuntimeError("Call reset() first to select a task.")
        return self._task_id

    def reset(self, task_index: int) -> tuple[str, dict[str, Any]]:
        from tau2.gym.gym_agent import AgentGymEnv

        self._task_id = self._task_index.task_id(task_index)
        self._env = AgentGymEnv(
            domain=self.cfg.domain,
            task_id=self._task_id,
            max_steps=self.cfg.max_steps,
            solo_mode=False,
            user_llm=self.cfg.user_llm,
            user_llm_args=self.cfg.user_llm_args,
            all_messages_as_observation=self.cfg.all_messages_as_observation,
        )
        observation, info = self._env.reset()

        tools = info.get("tools", [])
        self._tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]

        self._policy = info.get("policy", "")

        return observation, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        if self._env is None:
            raise RuntimeError("Call reset() before step().")
        return self._env.step(action)


def parse_reward_info(info: dict[str, Any]) -> dict[str, Any]:
    return parse_reward_info_value(info.get("reward_info"))


def parse_reward_info_value(reward_info: Any) -> dict[str, Any]:
    if reward_info is None:
        return {}
    if isinstance(reward_info, dict):
        return reward_info
    if isinstance(reward_info, str):
        try:
            parsed = json.loads(reward_info)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True, slots=True)
class PartialScoreWeights:
    action: float = 0.5
    communicate: float = 0.15
    env_assertion: float = 0.35
    db: float = 0.0


def compute_partial_score_from_reward_info(
    reward_info: dict[str, Any],
    *,
    weights: PartialScoreWeights = PartialScoreWeights(),
    normalize_over_present: bool = True,
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}

    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        matched = sum(1 for ac in action_checks if ac.get("action_match"))
        components["action"] = matched / len(action_checks)

    communicate_checks = reward_info.get("communicate_checks") or []
    if communicate_checks:
        met = sum(1 for cc in communicate_checks if cc.get("met"))
        components["communicate"] = met / len(communicate_checks)

    env_assertions = reward_info.get("env_assertions") or []
    if env_assertions:
        met = sum(1 for ea in env_assertions if ea.get("met"))
        components["env_assertion"] = met / len(env_assertions)

    db_check = reward_info.get("db_check") or {}
    if isinstance(db_check, dict) and "db_match" in db_check:
        components["db"] = 1.0 if db_check.get("db_match") else 0.0

    if not components:
        return 0.0, {}

    weight_map = {
        "action": weights.action,
        "communicate": weights.communicate,
        "env_assertion": weights.env_assertion,
        "db": weights.db,
    }

    if normalize_over_present:
        present = {k: weight_map[k] for k in components.keys()}
        weight_sum = sum(present.values())
        if weight_sum <= 0:
            return 0.0, components
        score = sum(components[k] * present[k] for k in components.keys()) / weight_sum
        return float(score), components

    score = sum(components[k] * weight_map[k] for k in components.keys())
    return float(score), components
