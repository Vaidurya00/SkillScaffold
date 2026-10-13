"""Async bridge for Tau2 Gym calls that block on the user simulator."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable

_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: ThreadPoolExecutor | None = None


def _get_tau2_env_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            workers = int(os.environ.get("TAU2_ENV_CONCURRENCY", "32"))
            if workers < 1:
                raise ValueError(f"TAU2_ENV_CONCURRENCY must be >= 1 (got {workers})")
            _EXECUTOR = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="tau2-env",
            )
    return _EXECUTOR


async def run_tau2_env_call(
    function: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a synchronous Tau2 Gym call without blocking asyncio rollouts."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_tau2_env_executor(),
        partial(function, *args, **kwargs),
    )


__all__ = ["run_tau2_env_call"]
