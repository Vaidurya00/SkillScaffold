"""Runtime registration for external Tau2 task sets."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable


CUSTOM_DOMAIN_PREFIX = "apigen_"

_LOCK = threading.Lock()
_REGISTERED: set[Path] = set()


def base_domain_for(domain: str) -> str:
    """Map a custom task-set domain back to the real Tau2 environment domain."""
    if domain.startswith(CUSTOM_DOMAIN_PREFIX):
        return domain[len(CUSTOM_DOMAIN_PREFIX) :]
    if domain.startswith("areal_"):
        parts = domain.split("_")
        if len(parts) >= 3 and parts[1] in {"airline", "retail", "telecom"}:
            return parts[1]
    return domain


def _make_snapshot_env_constructor(
    base_domain: str,
    db_file: str,
    registered_domain: str,
    task=None,
) -> Callable:
    """Build an environment pinned to one AReaL DB snapshot."""
    if base_domain == "airline":
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.airline.environment import get_environment

        def constructor(solo_mode: bool = False):
            environment = get_environment(
                db=FlightDB.load(db_file), solo_mode=solo_mode
            )
            environment.domain_name = registered_domain
            return environment

    elif base_domain == "retail":
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.retail.environment import get_environment

        def constructor(solo_mode: bool = False):
            environment = get_environment(
                db=RetailDB.load(db_file), solo_mode=solo_mode
            )
            environment.domain_name = registered_domain
            return environment

    elif base_domain == "telecom":
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.environment import get_environment

        def constructor(solo_mode: bool = False):
            environment = get_environment(
                db=TelecomDB.load(db_file), solo_mode=solo_mode
            )
            environment.domain_name = registered_domain
            return environment

    elif base_domain == "banking_knowledge":
        import os

        from tau2.domains.banking_knowledge.data_model import TransactionalDB
        from tau2.domains.banking_knowledge.environment import get_environment

        def constructor(solo_mode: bool = False):
            environment = get_environment(
                db=TransactionalDB.load(db_file),
                retrieval_variant=os.environ.get(
                    "TAU2_BANKING_RETRIEVAL_VARIANT", "bm25"
                ),
                task=task,
                solo_mode=solo_mode,
            )
            environment.domain_name = registered_domain
            return environment

    else:
        raise ValueError(f"Unsupported snapshot base domain: {base_domain}")

    return constructor


def _make_tasks_loader(domain_dir: Path):
    from tau2.data_model.tasks import Task

    tasks_cache: list[Task] | None = None
    splits_cache: dict[str, list[str]] | None = None

    def load_tasks() -> list[Task]:
        nonlocal tasks_cache
        if tasks_cache is None:
            with (domain_dir / "tasks.json").open(encoding="utf-8") as f:
                raw_tasks = json.load(f)
            tasks_cache = [Task.model_validate(task) for task in raw_tasks]
        return tasks_cache

    def load_splits() -> dict[str, list[str]]:
        nonlocal splits_cache
        if splits_cache is None:
            with (domain_dir / "split_tasks.json").open(encoding="utf-8") as f:
                splits_cache = json.load(f)
        return splits_cache

    def get_tasks(task_split_name: str | None = "base") -> list[Task]:
        tasks = load_tasks()
        if task_split_name is None:
            return tasks
        splits = load_splits()
        if (
            task_split_name == "base"
            and "base" not in splits
            and "train" in splits
        ):
            # AgentGymEnv calls the task loader without an explicit split.
            # Older generated training pools only contained "train".
            task_split_name = "train"
        if task_split_name not in splits:
            raise ValueError(
                f"Invalid task split name: {task_split_name}. "
                f"Valid splits are: {list(splits.keys())}"
            )
        wanted = set(splits[task_split_name])
        return [task for task in tasks if task.id in wanted]

    def get_task_splits() -> dict[str, list[str]]:
        return load_splits()

    return get_tasks, get_task_splits


def ensure_custom_tau2_registry(tasks_dir: str | Path | None) -> None:
    """Register APIGen task sets and manifest-defined DB snapshot task sets.

    APIGen airline/retail reuse the base environment. APIGen banking pins its
    retrieval variant. A manifest maps each external task-set name to a base
    domain and DB file, and those domains get a fresh DB instance per episode.
    """
    if not tasks_dir:
        return

    root = Path(tasks_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"TAU2_CUSTOM_TASKS_DIR does not exist: {root}")

    with _LOCK:
        if root in _REGISTERED:
            return

        from tau2.registry import registry

        for base_domain in ("airline", "retail", "banking_knowledge"):
            domain_dir = root / base_domain
            if not (domain_dir / "tasks.json").exists():
                continue
            if not (domain_dir / "split_tasks.json").exists():
                continue

            custom_domain = f"{CUSTOM_DOMAIN_PREFIX}{base_domain}"
            if custom_domain not in registry.get_domains():
                base_env_constructor: Callable = registry.get_env_constructor(base_domain)
                if base_domain == "banking_knowledge":
                    env_constructor = _banking_env_constructor(base_env_constructor)
                else:
                    env_constructor = base_env_constructor
                registry.register_domain(env_constructor, custom_domain)

            if custom_domain not in registry.get_task_sets():
                get_tasks, get_task_splits = _make_tasks_loader(domain_dir)
                registry.register_tasks(
                    get_tasks,
                    custom_domain,
                    get_task_splits=get_task_splits,
                )

        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as f:
                manifest = json.load(f)
            for registered_domain, info in manifest.items():
                domain_dir = root / registered_domain
                if not (domain_dir / "tasks.json").exists():
                    raise FileNotFoundError(
                        f"Manifest task set is missing tasks.json: {domain_dir}"
                    )
                if not (domain_dir / "split_tasks.json").exists():
                    raise FileNotFoundError(
                        f"Manifest task set is missing split_tasks.json: {domain_dir}"
                    )

                base_domain = str(info["domain"])
                db_file = str(info["db_file"])
                if not Path(db_file).is_file():
                    raise FileNotFoundError(
                        f"Manifest DB snapshot does not exist: {db_file}"
                    )
                if registered_domain not in registry.get_domains():
                    raw_tasks = json.loads(
                        (domain_dir / "tasks.json").read_text(encoding="utf-8")
                    )
                    registered_task = None
                    if len(raw_tasks) == 1:
                        from tau2.data_model.tasks import Task

                        registered_task = Task.model_validate(raw_tasks[0])
                    registry.register_domain(
                        _make_snapshot_env_constructor(
                            base_domain=base_domain,
                            db_file=db_file,
                            registered_domain=registered_domain,
                            task=registered_task,
                        ),
                        registered_domain,
                    )
                if registered_domain not in registry.get_task_sets():
                    get_tasks, get_task_splits = _make_tasks_loader(domain_dir)
                    registry.register_tasks(
                        get_tasks,
                        registered_domain,
                        get_task_splits=get_task_splits,
                    )

        _REGISTERED.add(root)


def _banking_env_constructor(base_constructor: Callable) -> Callable:
    """Wrap the banking_knowledge env constructor to pin the retrieval variant.

    The banking env takes a `retrieval_variant` kwarg the other domains don't
    have. RL rollouts must use one frozen variant (train/eval consistency), so
    we read it once from the environment:

        TAU2_BANKING_RETRIEVAL_VARIANT  (default: "qwen_embeddings")

    For qwen_embeddings, the embedder client also needs:
        OPENROUTER_BASE_URL = http://<host>:8010/v1   (local vLLM Qwen3-Embedding-8B)
        OPENROUTER_API_KEY  = EMPTY
    """
    import os

    variant = os.environ.get("TAU2_BANKING_RETRIEVAL_VARIANT", "qwen_embeddings")

    def constructor(**kwargs):
        kwargs.setdefault("retrieval_variant", variant)
        return base_constructor(**kwargs)

    return constructor
