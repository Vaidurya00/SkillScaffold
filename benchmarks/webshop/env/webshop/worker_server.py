"""HTTP worker for the full WebShop text environment.

Each process loads the product catalogue and Lucene searcher once, then shares
that immutable server across lightweight per-episode browser sessions.  This
keeps the memory footprint bounded while still allowing independent sessions
for concurrent Slime rollouts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, request

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

app = Flask(__name__)
app.logger.disabled = True
logging.getLogger("werkzeug").setLevel(logging.ERROR)

_worker_id = "worker"
_env_cls = None
_shared_server = None
_env_kwargs: dict = {}
_episodes: dict[str, dict] = {}
_episodes_lock = threading.Lock()
_env_create_lock = threading.Lock()
_idle_timeout = max(
    60, int(os.environ.get("WEBSHOP_EPISODE_IDLE_TIMEOUT_SECONDS", str(2 * 60 * 60)))
)


def _available_actions(env) -> list[str]:
    available = env.get_available_actions() or {}
    actions: list[str] = []
    if available.get("has_search_bar"):
        actions.append("search[<keywords>]")
    actions.extend(
        f"click[{value}]"
        for value in available.get("clickables", [])
        if str(value).strip()
    )
    return actions


def _new_env(goal_idx: int):
    # WebAgentTextEnv.__init__ calls reset() *without forwarding* the
    # constructor's ``session`` kwarg. Passing session=goal_idx to the
    # constructor therefore silently creates a random goal.  In addition,
    # using str(goal_idx) as the browser session id makes concurrent episodes
    # for the same goal share mutable SimServer state.
    #
    # Construct the lightweight browser, discard its unavoidable random
    # bootstrap session, then reproduce WebAgentTextEnv.reset with a unique
    # browser session id while passing goal_idx separately as session_int.
    # The upstream constructor also resets process-global RNG state, so two
    # simultaneous constructors generate the same bootstrap session id. Keep
    # only this short construction/reset section serialized per worker.
    with _env_create_lock:
        env = _env_cls(server=_shared_server, **_env_kwargs)
        bootstrap_session = str(env.session)
        episode_session = f"goal-{int(goal_idx)}-{uuid.uuid4().hex}"
        init_url = f"{env.base_url}/{episode_session}"
        env.browser.get(
            init_url,
            session_id=episode_session,
            session_int=int(goal_idx),
        )
        env.session = episode_session
        env.text_to_clickable = None
        env.instruction_text = env.get_instruction_text()
        obs = str(env.observation)
        env.prev_obs = [obs]
        env.prev_actions = []

        # The constructor bootstrap is never used again. Removing it prevents
        # one leaked shared-server session per reset.
        if bootstrap_session != episode_session:
            _shared_server.user_sessions.pop(bootstrap_session, None)

        expected = str(_shared_server.goals[int(goal_idx)]["instruction_text"])
        actual = str(env.instruction_text)
        if actual.startswith("Instruction:"):
            actual = actual[len("Instruction:") :].strip()
        if actual != expected:
            _shared_server.user_sessions.pop(episode_session, None)
            raise RuntimeError(
                "WebShop goal mismatch after reset: "
                f"goal_idx={goal_idx}, expected={expected!r}, actual={actual!r}"
            )
        return env, obs


def _pop_episode(episode_id: str):
    with _episodes_lock:
        return _episodes.pop(episode_id, None)


def _dispose_episode(episode: dict | None) -> None:
    if not episode:
        return
    env = episode.get("env")
    if env is None:
        return
    # WebAgentTextEnv.step() calls self.reset() automatically after a terminal
    # purchase, which changes env.session before the worker sees done=True.
    # Retain and delete the immutable episode session as well as any terminal
    # reset session created by that upstream behavior.
    session_ids = {
        str(episode.get("server_session_id") or ""),
        str(getattr(env, "session", "") or ""),
    }
    try:
        env.close()
    finally:
        for session_id in session_ids:
            if session_id:
                _shared_server.user_sessions.pop(session_id, None)


@app.get("/health")
def health():
    with _episodes_lock:
        active = len(_episodes)
    return jsonify(
        status="ok",
        worker_id=_worker_id,
        active_episodes=active,
        server_sessions=len(_shared_server.user_sessions) if _shared_server is not None else 0,
        goals=len(_shared_server.goals) if _shared_server is not None else 0,
    )


@app.post("/reset")
def reset():
    payload = request.get_json(force=True) or {}
    if "goal_idx" not in payload:
        return jsonify(error="goal_idx is required"), 400

    try:
        goal_idx = int(payload["goal_idx"])
        if goal_idx < 0 or goal_idx >= len(_shared_server.goals):
            return jsonify(error=f"goal_idx out of range: {goal_idx}"), 400

        env, obs = _new_env(goal_idx)
        episode_id = str(uuid.uuid4())
        # The rendered page prefixes this with "Instruction:". Use the
        # underlying goal text so it is byte-identical to the parquet/cache
        # retrieval key.
        task_description = str(
            _shared_server.goals[goal_idx]["instruction_text"]
        )
        episode = {
            "env": env,
            "server_session_id": str(env.session),
            "goal_idx": goal_idx,
            "step": 0,
            "lock": threading.Lock(),
            "last_used_at": time.monotonic(),
        }
        with _episodes_lock:
            _episodes[episode_id] = episode

        return jsonify(
            episode_id=episode_id,
            obs=obs,
            available_actions=_available_actions(env),
            task_description=task_description,
            goal_idx=goal_idx,
            step=0,
            worker_id=_worker_id,
        )
    except Exception as exc:
        return jsonify(
            error=str(exc),
            error_type=type(exc).__name__,
            worker_id=_worker_id,
        ), 500


@app.post("/step")
def step():
    payload = request.get_json(force=True) or {}
    episode_id = str(payload.get("episode_id") or "")
    action = str(payload.get("action") or "")

    with _episodes_lock:
        episode = _episodes.get(episode_id)
    if episode is None:
        return jsonify(error=f"Unknown episode_id: {episode_id}"), 404

    with episode["lock"]:
        env = episode["env"]
        try:
            before = set(_available_actions(env))
            action_is_valid = (
                action.startswith("search[")
                and action.endswith("]")
                and len(action) > len("search[]")
            ) or action in before

            # The upstream WebShop environment raises Python exceptions for
            # many invalid clicks instead of returning an observation.  Treat
            # those as ordinary agent mistakes: preserve the page and give
            # explicit feedback so the next policy step can recover.
            if not action_is_valid:
                episode["step"] += 1
                episode["last_used_at"] = time.monotonic()
                choices = sorted(before)
                obs = (
                    f"Invalid action: {action!r}. "
                    f"Choose one of the admissible actions: {choices}\n"
                    f"Current page: {env.observation}"
                )
                return jsonify(
                    episode_id=episode_id,
                    obs=obs,
                    available_actions=choices,
                    reward=0.0,
                    task_score=0.0,
                    won=False,
                    done=False,
                    valid_action=False,
                    goal_idx=episode["goal_idx"],
                    step=episode["step"],
                    worker_id=_worker_id,
                )

            obs, score, done, _ = env.step(action)
            episode["step"] += 1
            episode["last_used_at"] = time.monotonic()
            score = float(score or 0.0)
            won = bool(done and score >= 1.0 - 1e-8)

            result = dict(
                episode_id=episode_id,
                obs=str(obs),
                available_actions=[] if done else _available_actions(env),
                reward=1.0 if won else 0.0,
                task_score=score,
                won=won,
                done=bool(done),
                valid_action=bool(action_is_valid),
                goal_idx=episode["goal_idx"],
                step=episode["step"],
                worker_id=_worker_id,
            )
            if done:
                _dispose_episode(_pop_episode(episode_id))
            return jsonify(result)
        except Exception as exc:
            # A syntactically valid action can still expose an upstream page
            # edge case (for example, a missing option value).  Keep this
            # recoverable as well; HTTP 5xx would make Slime retry the same
            # broken action until the rollout times out.
            episode["step"] += 1
            episode["last_used_at"] = time.monotonic()
            choices = _available_actions(env)
            obs = (
                f"Action failed: {action!r} ({type(exc).__name__}: {exc}). "
                f"Choose another admissible action: {choices}\n"
                f"Current page: {env.observation}"
            )
            return jsonify(
                episode_id=episode_id,
                obs=obs,
                available_actions=choices,
                reward=0.0,
                task_score=0.0,
                won=False,
                done=False,
                valid_action=False,
                goal_idx=episode["goal_idx"],
                step=episode["step"],
                worker_id=_worker_id,
            )


@app.post("/close")
def close():
    payload = request.get_json(force=True) or {}
    episode_id = str(payload.get("episode_id") or "")
    episode = _pop_episode(episode_id)
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
                episode_id
                for episode_id, episode in _episodes.items()
                if episode["last_used_at"] < cutoff
            ]
        for episode_id in stale:
            _dispose_episode(_pop_episode(episode_id))


def main():
    global _worker_id, _env_cls, _shared_server, _env_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--vendor-root", required=True)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--attr-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _worker_id = args.worker_id
    vendor_root = os.path.abspath(args.vendor_root)
    sys.path.insert(0, vendor_root)

    import gym
    from web_agent_site.envs import WebAgentTextEnv

    _env_cls = WebAgentTextEnv
    _env_kwargs = {
        "observation_mode": "text",
        "file_path": os.path.abspath(args.file_path),
        "attr_path": os.path.abspath(args.attr_path),
        "num_products": None,
        "human_goals": True,
        "seed": args.seed,
    }

    # Construct one environment to initialize the shared catalogue/searcher.
    prototype = gym.make(
        "WebAgentTextEnv-v0",
        disable_env_checker=True,
        **_env_kwargs,
    )
    _shared_server = prototype.server
    prototype_session = str(getattr(prototype, "session", "") or "")
    prototype.close()
    if prototype_session:
        _shared_server.user_sessions.pop(prototype_session, None)

    threading.Thread(target=_idle_cleanup_loop, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
