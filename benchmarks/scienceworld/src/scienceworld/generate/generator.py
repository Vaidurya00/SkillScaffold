"""ScienceWorld-specific wrappers around the shared policy Generator."""

from __future__ import annotations

from src.scienceworld.prompts import (
    format_successful_scout_generator_messages,
    skill_json_to_text,
)
from src.webshop.generate.generator import generate_generator_sample


def resolve_task_description(label: dict | None) -> str:
    if not isinstance(label, dict):
        return ""
    return str(label.get("task_description") or "")


def parse_successful_scout_skill_output(args, response_text: str) -> dict:
    skill_text, error, word_count = skill_json_to_text(
        response_text,
        min_words=20,
        max_words=200,
        min_sentences=1,
        max_sentences=6,
    )
    penalty = abs(float(getattr(args, "skill_format_penalty", 1.0)))
    if skill_text is None:
        return {
            "skill": "",
            "raw_skill_output": str(response_text or "").strip(),
            "skill_word_count": int(word_count),
            "skill_schema": "scienceworld_card_v1",
            "format_penalty": -penalty,
            "format_score": 0.0,
            "skill_parse_error": error,
        }
    headings = (
        "# Skill:",
        "## When to Use",
        "## Inputs to Identify",
        "## Workflow",
        "## Decision Rules",
    )
    positions = [skill_text.find(heading) for heading in headings]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return {
            "skill": "",
            "raw_skill_output": str(response_text or "").strip(),
            "skill_word_count": int(word_count),
            "skill_schema": "scienceworld_card_v1",
            "format_penalty": -penalty,
            "format_score": 0.0,
            "skill_parse_error": "missing or misordered canonical card headings",
        }
    return {
        "skill": skill_text,
        "skill_word_count": int(word_count),
        "skill_schema": "scienceworld_card_v1",
        "format_penalty": 0.0,
        "format_score": 1.0,
    }


async def build_successful_scout_generator_sample(
    args,
    *,
    task_description: str,
    trajectory: str,
    sampling_params: dict,
    n_idx: int,
    candidate_count: int,
    parser_fn=None,
    base_metadata: dict | None = None,
):
    if candidate_count != 4 or not 0 <= n_idx < candidate_count:
        raise ValueError(
            "Direct-Delta requires four candidate indices 0..3, got "
            f"index={n_idx}, count={candidate_count}"
        )
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "generator_candidate_index": int(n_idx),
            "generator_candidate_count": int(candidate_count),
            "generator_optimizer_enabled": 0.0,
            "training_stage": "scienceworld_successful_scout_extraction",
        }
    )
    return await generate_generator_sample(
        args,
        messages=format_successful_scout_generator_messages(
            task_description, trajectory
        ),
        sampling_params=sampling_params,
        n_idx=n_idx,
        requested_mode="scienceworld_successful_scout_v1",
        actual_mode="scienceworld_successful_scout_v1",
        parser_fn=parser_fn or parse_successful_scout_skill_output,
        base_metadata=metadata,
    )
