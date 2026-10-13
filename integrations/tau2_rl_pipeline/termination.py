"""Helpers for preserving Tau2's native simulation termination reason."""

from __future__ import annotations

import json
from typing import Any


TAU2_TERMINATION_REASONS = (
    "user_stop",
    "agent_stop",
    "max_steps",
    "timeout",
    "too_many_errors",
    "agent_error",
    "user_error",
    "infrastructure_error",
    "context_window_exceeded",
    "unexpected_error",
)


def tau2_termination_reason_from_info(info: dict[str, Any] | None) -> str | None:
    """Return Tau2's native SimulationRun termination reason, if available."""
    if not isinstance(info, dict):
        return None
    simulation_run = info.get("simulation_run")
    if isinstance(simulation_run, str):
        try:
            simulation_run = json.loads(simulation_run)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(simulation_run, dict):
        return None
    reason = simulation_run.get("termination_reason")
    return reason.strip() if isinstance(reason, str) and reason.strip() else None
