"""Language-quality audit for WebShop generator skills."""

from __future__ import annotations

import re

from slime.utils.types import Sample

from src.utils.skill_audit import audit_generator_samples as _audit


_UNSAFE_POLICY_PATTERNS = (
    re.compile(
        r"\b(?:adjust|increase|raise|expand)\s+(?:the\s+)?"
        r"(?:budget|price\s+(?:limit|cap|range))\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bexpand\s+(?:your\s+|the\s+)?(?:[A-Za-z-]+\s+or\s+)?"
        r"(?:budget|price\s+(?:limit|cap|range))\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|exceed)\s+(?:the\s+)?"
        r"(?:budget|price\s+(?:limit|cap)|hard\s+constraint)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|buy|purchase|select|accept|use)\s+(?:the\s+)?"
        r"(?:nearest|closest|partial)\s+(?:match|alternative|option)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bsettle\s+for\b", flags=re.IGNORECASE),
)


def _unsafe_policy_reason(skill_text: str) -> str:
    for pattern in _UNSAFE_POLICY_PATTERNS:
        match = pattern.search(str(skill_text or ""))
        if match is not None:
            return f"unsafe hard-constraint relaxation: {match.group(0)!r}"
    return ""


async def audit_generator_samples(args, generator_samples: list[Sample]) -> None:
    await _audit(args, generator_samples, domain="webshop")
    for sample in generator_samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        reason = _unsafe_policy_reason(str(metadata.get("skill") or ""))
        metadata["skill_policy_safety_violation"] = 1.0 if reason else 0.0
        metadata["skill_policy_safety_reason"] = reason
        if not reason:
            continue
        metadata["skill_language_audit_applied"] = 1.0
        metadata["skill_language_audit_pass"] = 0.0
        metadata["skill_language_audit_reward_zeroed"] = 1.0
        metadata["skill_language_audit_status"] = "failed"
        metadata["skill_language_audit_judgment"] = "FAIL"
        prior_reason = str(metadata.get("skill_language_audit_reason") or "")
        metadata["skill_language_audit_reason"] = (
            f"{prior_reason} | {reason}" if prior_reason else reason
        )
