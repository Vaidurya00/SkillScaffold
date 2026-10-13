"""Read-only SkillRL-compatible template retrieval for Tau2 rollouts.

This module intentionally implements only the static-bank inference path.  It
does not update, save, or resume skills, and therefore cannot alter rewards or
GRPO advantages.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any


_LOCK = threading.Lock()
_MEMORY: "StaticSkillMemory | None" = None


class StaticSkillMemory:
    """The template-mode subset of SkillRL's ``SkillsOnlyMemory``."""

    def __init__(self, path: str, task_specific_top_k: int | None = None):
        with open(path, encoding="utf-8") as handle:
            self.skills = json.load(handle)
        self.task_specific_top_k = task_specific_top_k
        n_general = len(self.skills.get("general_skills", []))
        n_task = sum(
            len(items)
            for items in self.skills.get("task_specific_skills", {}).values()
        )
        n_mistakes = len(self.skills.get("common_mistakes", []))
        print(
            f"[StaticSkillMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes "
            "| retrieval_mode=template | updates=disabled"
        )

    def _detect_task_type(self, task_description: str) -> str:
        task_specific = self.skills.get("task_specific_skills", {})
        goal = task_description.lower()
        for key in task_specific:
            if key.lower() in goal:
                return key
        return "unknown"

    def retrieve(self, task_description: str, top_k: int = 6) -> dict[str, Any]:
        task_type = self._detect_task_type(task_description)
        all_general = self.skills.get("general_skills", [])
        dynamic_skills = [
            skill
            for skill in all_general
            if skill.get("skill_id", "").startswith("dyn_")
        ]
        static_skills = [
            skill
            for skill in all_general
            if not skill.get("skill_id", "").startswith("dyn_")
        ]
        general_skills = dynamic_skills + static_skills[
            : max(0, top_k - len(dynamic_skills))
        ]
        task_skills = self.skills.get("task_specific_skills", {}).get(
            task_type, []
        )
        if self.task_specific_top_k is not None:
            task_skills = task_skills[: self.task_specific_top_k]
        return {
            "general_skills": general_skills,
            "task_specific_skills": task_skills,
            "mistakes_to_avoid": self.skills.get("common_mistakes", [])[:5],
            "task_type": task_type,
            "task_specific_examples": [],
            "retrieval_mode": "template",
        }

    @staticmethod
    def format_for_prompt(retrieved: dict[str, Any]) -> str:
        sections: list[str] = []
        task_type = retrieved.get("task_type", "unknown")

        general_skills = retrieved.get("general_skills", [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                lines.append(
                    f"- **{skill.get('title', '')}**: "
                    f"{skill.get('principle', '')}"
                )
            sections.append("\n".join(lines))

        task_skills = retrieved.get("task_specific_skills", [])
        if task_skills:
            task_name = task_type.replace("_", " ").title()
            lines = [f"### {task_name} Skills"]
            for skill in task_skills:
                lines.append(
                    f"- **{skill.get('title', '')}**: "
                    f"{skill.get('principle', '')}"
                )
                when = skill.get("when_to_apply", "")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        mistakes = retrieved.get("mistakes_to_avoid", [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                description = mistake.get("description", "")
                fix = mistake.get("how_to_avoid", "")
                if description:
                    lines.append(f"- **Don't**: {description}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return (
            "\n\n".join(sections)
            if sections
            else "No relevant skills found for this task."
        )


def get_static_skill_block(domain: str, opening_user_message: str) -> str:
    """Return the fixed SkillRL prompt extension, or empty when disabled."""
    global _MEMORY
    if os.environ.get("TAU2_STATIC_SKILLS_ENABLED", "0") != "1":
        return ""
    if _MEMORY is None:
        with _LOCK:
            if _MEMORY is None:
                path = os.environ.get("TAU2_STATIC_SKILLS_JSON", "")
                if not path or not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"TAU2_STATIC_SKILLS_JSON does not exist: {path!r}"
                    )
                _MEMORY = StaticSkillMemory(
                    path,
                    task_specific_top_k=int(
                        os.environ.get("TAU2_STATIC_SKILLS_TASK_TOP_K", "6")
                    ),
                )
    retrieved = _MEMORY.retrieve(
        f"[{domain}] {opening_user_message or ''}",
        top_k=int(os.environ.get("TAU2_STATIC_SKILLS_TOP_K", "6")),
    )
    text = _MEMORY.format_for_prompt(retrieved)
    if not text or text.startswith("No relevant skills"):
        return ""
    return f"## Retrieved Relevant Experience\n{text}\n\n"
