"""HTTP worker hosting independent ScienceWorld JVM episodes."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import uuid

from flask import Flask, jsonify, request


app = Flask(__name__)
app.logger.disabled = True
logging.getLogger("werkzeug").setLevel(logging.ERROR)

_worker_id = "worker"
_simplification = "easy"
_default_max_steps = 30
_episodes: dict[str, dict] = {}
_episodes_lock = threading.Lock()
_idle_timeout = max(
    60, int(os.environ.get("SCIENCEWORLD_EPISODE_IDLE_TIMEOUT_SECONDS", "7200"))
)


def _action_hints(env) -> str:
    actions = [str(item) for item in env.get_possible_actions()]
    objects = [str(item) for item in env.get_possible_objects()]
    return (
        "Action templates: " + repr(actions) + "\n"
        "Observable object referents for OBJ: " + repr(objects) + "\n"
        "Replace OBJ exactly with one observable referent; output one command only."
    )


def _valid_action_response(obs: str, info: dict) -> bool:
    for key in ("valid", "isValid", "actionValid", "action_valid"):
        if key in info:
            return bool(info[key])
    lowered = str(obs or "").strip().lower()
    invalid_prefixes = (
        "no known action matches",
        "unknown action",
        "invalid action",
        "i don't understand",
    )
    return not lowered.startswith(invalid_prefixes)


def _new_env(task_name: str, variation_idx: int, split: str, max_steps: int):
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv(envStepLimit=max_steps)
    try:
        env.load(task_name, 0, _simplification)
        getter = getattr(env, f"get_variations_{split}", None)
        if getter is None:
            raise ValueError(f"Unsupported ScienceWorld split: {split}")
        allowed = {int(item) for item in getter()}
        if variation_idx not in allowed:
            raise ValueError(
                f"Variation {variation_idx} is not in {task_name}/{split}"
            )
        env.envStepLimit = max_steps
        env.load(task_name, variation_idx, _simplification)
        obs, info = env.reset()
        return env, str(obs), dict(info or {})
    except Exception:
        env.close()
        raise


def _pop_episode(episode_id: str):
    with _episodes_lock:
        return _episodes.pop(episode_id, None)


def _dispose_episode(episode: dict | None) -> None:
    if not episode:
        return
    try:
        episode["env"].close()
    except Exception:
        pass


@app.get("/health")
def health():
    with _episodes_lock:
        active = len(_episodes)
    return jsonify(status="ok", worker_id=_worker_id, active_episodes=active)


@app.post("/reset")
def reset():
    payload = request.get_json(force=True) or {}
    required = ("task_name", "variation_idx", "split")
    missing = [key for key in required if key not in payload]
    if missing:
        return jsonify(error=f"missing reset fields: {missing}"), 400
    try:
        task_name = str(payload["task_name"])
        variation_idx = int(payload["variation_idx"])
        split = str(payload["split"])
        max_steps = int(payload.get("max_steps", _default_max_steps))
        env, obs, info = _new_env(task_name, variation_idx, split, max_steps)
        episode_id = str(uuid.uuid4())
        episode = {
            "env": env,
            "lock": threading.Lock(),
            "step": 0,
            "last_used_at": time.monotonic(),
            "task_name": task_name,
            "variation_idx": variation_idx,
            "split": split,
        }
        with _episodes_lock:
            _episodes[episode_id] = episode
        task_description = str(env.get_task_description()).strip()
        return jsonify(
            episode_id=episode_id,
            obs=obs,
            action_hints=_action_hints(env),
            task_description=task_description,
            task_name=task_name,
            variation_idx=variation_idx,
            split=split,
            step=0,
            task_score=float(info.get("score", 0.0) or 0.0),
            worker_id=_worker_id,
        )
    except Exception as exc:
        return jsonify(
            error=str(exc), error_type=type(exc).__name__, worker_id=_worker_id
        ), 500


@app.post("/step")
def step():
    payload = request.get_json(force=True) or {}
    episode_id = str(payload.get("episode_id") or "")
    action = str(payload.get("action") or "").strip()
    with _episodes_lock:
        episode = _episodes.get(episode_id)
    if episode is None:
        return jsonify(error=f"Unknown episode_id: {episode_id}"), 404

    with episode["lock"]:
        try:
            obs, raw_reward, done, info = episode["env"].step(action)
            info = dict(info or {})
            episode["step"] += 1
            episode["last_used_at"] = time.monotonic()
            task_score = float(info.get("score", 0.0) or 0.0)
            won = task_score >= 100.0
            terminal = bool(done or won)
            result = dict(
                episode_id=episode_id,
                obs=str(obs),
                action_hints="" if terminal else _action_hints(episode["env"]),
                reward=1.0 if won else 0.0,
                raw_reward=float(raw_reward or 0.0),
                task_score=task_score,
                won=won,
                done=terminal,
                valid_action=_valid_action_response(str(obs), info),
                step=episode["step"],
                worker_id=_worker_id,
            )
            if terminal:
                _dispose_episode(_pop_episode(episode_id))
            return jsonify(result)
        except Exception as exc:
            return jsonify(
                error=str(exc),
                error_type=type(exc).__name__,
                episode_id=episode_id,
                worker_id=_worker_id,
            ), 500


@app.post("/close")
def close():
    payload = request.get_json(force=True) or {}
    episode = _pop_episode(str(payload.get("episode_id") or ""))
    if episode is None:
        return jsonify(success=True, already_closed=True)
    _dispose_episode(episode)
    return jsonify(success=True)


def _idle_cleanup_loop():
    while True:
        time.sleep(60)
        cutoff = time.monotonic() - _idle_timeout
        with _episodes_lock:
            stale = [
                key
                for key, episode in _episodes.items()
                if episode["last_used_at"] < cutoff
            ]
        for episode_id in stale:
            _dispose_episode(_pop_episode(episode_id))


def main():
    global _worker_id, _simplification, _default_max_steps
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--simplification", default="easy")
    parser.add_argument("--default-max-steps", default=30, type=int)
    args = parser.parse_args()
    _worker_id = args.worker_id
    _simplification = args.simplification
    _default_max_steps = args.default_max_steps
    threading.Thread(target=_idle_cleanup_loop, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
