"""Prompts for no-native-think ScienceWorld Delta-card training."""

from __future__ import annotations

import json


_OUTPUT_CONTRACT = """At every step, return exactly one concise reasoning-and-action pair and no other text:
<reasoning>reason about only the next action in at most 120 English words</reasoning>
<action>write exactly one ScienceWorld command using the current action templates and observable objects</action>
Use each tag exactly once and stop immediately after </action>."""


EXECUTOR_SYSTEM_PROMPT = f"""You are an expert agent operating in ScienceWorld, a simulated science laboratory.
Complete the assigned task using only facts in the task, observations, and current action/object hints. Treat ScienceWorld commands as free-form templates: replace OBJ with an observable object name, execute one command per turn, and use the resulting observation and score as feedback. Do not assume an experiment succeeded until the environment provides observable evidence. Prefer short, reversible information-gathering actions when evidence is missing.

{_OUTPUT_CONTRACT}"""


EXECUTOR_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE = f"""You are an expert agent operating in ScienceWorld, a simulated science laboratory.
Complete the assigned task using only facts in the task, observations, and current action/object hints. Treat ScienceWorld commands as free-form templates: replace OBJ with an observable object name, execute one command per turn, and use the resulting observation and score as feedback. Do not assume an experiment succeeded until the environment provides observable evidence.

A candidate reusable skill is provided below. Silently decide whether its observable trigger matches the current state before each action. Apply it only when relevant; otherwise ignore it. The current task, observations, and action/object hints always take priority. The skill is strategy guidance, never evidence about an object or experiment outcome.

Reference Skill:
{{skill}}

{_OUTPUT_CONTRACT}"""


def format_executor_messages(task_description: str) -> list[dict]:
    return [
        {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]


def format_executor_with_skill_messages(
    task_description: str, skill: str
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": EXECUTOR_WITH_SKILL_SYSTEM_PROMPT_TEMPLATE.format(skill=skill),
        },
        {"role": "user", "content": f"Your task is to: {task_description}"},
    ]


SUCCESSFUL_SCOUT_GENERATOR_SYSTEM_PROMPT = """You are the Skill Generator in a reinforcement-learning ScienceWorld agent. Given one trajectory that the environment confirms succeeded, extract exactly one reusable procedural skill grounded only in the supplied evidence. Success does not imply that every action was useful, necessary, or optimal."""


SUCCESSFUL_SCOUT_GENERATOR_USER_PROMPT_TEMPLATE = """Private task:
{task_description}

Private environment-confirmed successful trajectory:
<<<BEGIN_SUCCESSFUL_TRAJECTORY>>>
{trajectory}
<<<END_SUCCESSFUL_TRAJECTORY>>>

STRICT SERIALIZATION CONTRACT:
1. Output exactly one valid JSON object with exactly one key named `skill`.
2. Output no analysis, preface, code fence, or trailing text outside the JSON object.
3. The `skill` value must be a 50-150 word Markdown card with headings in this exact order: `# Skill:`, `## When to Use`, `## Inputs to Identify`, `## Workflow`, `## Decision Rules`.
4. Use no more than six complete sentences; concise list fragments are allowed.

Extract one reusable scientific decision mechanism supported by the evidence. State an observable trigger, the action templates and object properties to identify, an ordered experiment or navigation workflow, and an observable verification or stopping condition. Prefer mechanisms such as locating a target before manipulation, controlled comparison, conductivity testing, circuit completion, state-change verification, or recovery after an invalid command.

Abstract away task-specific object names, room names, colors, identifiers, and variation details. Do not copy a complete action sequence. Do not invent instruments, controls, object properties, or commands absent from the evidence. Do not mention the private task, trajectory, example, outcome, reward, or success. Do not claim universal optimality or treat plausibility as observed evidence.

Your response must begin with `{{"skill":"# Skill:` and remain valid JSON."""


def format_successful_scout_generator_messages(
    task_description: str, trajectory: str
) -> list[dict]:
    trajectory = str(trajectory or "").strip()
    if not trajectory:
        raise ValueError("Successful scout extraction requires a non-empty trajectory")
    return [
        {"role": "system", "content": SUCCESSFUL_SCOUT_GENERATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SUCCESSFUL_SCOUT_GENERATOR_USER_PROMPT_TEMPLATE.format(
                task_description=str(task_description or "").strip(),
                trajectory=trajectory,
            ),
        },
    ]


def skill_json_to_text(
    json_str: str,
    *,
    min_words: int = 20,
    max_words: int = 200,
    min_sentences: int = 1,
    max_sentences: int = 6,
) -> tuple[str | None, str, int]:
    """Reuse the reviewed strict JSON/card parser from the WebShop method."""

    from src.webshop.prompts import skill_json_to_text as parse

    return parse(
        json_str,
        min_words=min_words,
        max_words=max_words,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )


def dump_prompt_contract() -> str:
    """Small inspection helper used by the smoke tests."""

    return json.dumps(
        {
            "native_thinking": False,
            "executor_tags": ["reasoning", "action"],
            "skill_headings": [
                "# Skill:",
                "## When to Use",
                "## Inputs to Identify",
                "## Workflow",
                "## Decision Rules",
            ],
        },
        sort_keys=True,
    )
