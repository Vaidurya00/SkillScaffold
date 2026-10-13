"""Tau2-specific compact card extraction prompt."""

from __future__ import annotations


def format_tau_delta_card_messages(
    task_description: str,
    trajectory: str,
    won_score: float,
) -> list[dict[str, str]]:
    success = float(won_score) >= 0.9
    return [{
        "role": "user",
        "content": f"""Task:
{task_description}

Agent trajectory:
<<<BEGIN_TRAJECTORY>>>
{trajectory}
<<<END_TRAJECTORY>>>

True environment success: {success}

Extract exactly one reusable customer-service skill supported by this trajectory.
Concentrate on one mechanism, not a generic customer-service guide. Capture the
observable trigger, required information checks, policy/tool ordering, confirmation
boundary, and a concrete stopping or recovery condition. The Tau policy and tool
schemas always override this skill. Do not copy customer names, IDs, dates, account
numbers, or task-specific values.

Return only one valid JSON object with exactly this schema:
{{"skill": "a 50-180 word action-oriented reusable skill"}}
""",
    }]
