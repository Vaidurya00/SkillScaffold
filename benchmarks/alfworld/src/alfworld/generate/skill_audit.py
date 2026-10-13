"""Language-quality audit for ALFWorld generator skills."""

from __future__ import annotations

from slime.utils.types import Sample

from src.utils.skill_audit import audit_generator_samples as _audit


async def audit_generator_samples(args, generator_samples: list[Sample]) -> None:
    await _audit(args, generator_samples, domain="alfworld")
