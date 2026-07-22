"""Single-agent Gymnasium view of a 2-player game for self-play training.

The learner controls one seat; the other seat is played by a *frozen* model whose moves this wrapper
supplies automatically over the server's duel channel. SB3 only ever sees the learner's decision
points, and each step's reward (from the learner's perspective) sums the learner's move plus the
opponent's reply. Use with sb3-contrib MaskablePPO (exposes action_masks()).
"""
import json
import random
import subprocess
from collections.abc import Mapping

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ml_lab.controllers import (
    ControllerBinding,
    ControllerResolutionError,
    ControllerResolver,
    ResolvedController,
    predict as predict_resolved,
    validate_inference_input,
)


def bind_opponents(opponents, resolver: ControllerResolver) -> list[ControllerBinding]:
    """Bind self-play opponents once; raw models lack the metadata required for safe inference."""
    raw_opponents = list(opponents) if isinstance(opponents, (list, tuple)) else [opponents]
    bindings: list[ControllerBinding] = []
    for opponent in raw_opponents:
        if isinstance(opponent, ControllerBinding):
            bindings.append(opponent)
        elif isinstance(opponent, (str, Mapping)):
            bindings.append(resolver.bind(opponent))
        else:
            raise ControllerResolutionError(
                "self-play opponents must be controller specifications or ControllerBinding instances"
            )
    return bindings


class SelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, server_cmd, opponent_models, learner_seat: int = 0, base_seed: int = 0):
        super().__init__()
        self.proc = subprocess.Popen(list(server_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self.learner = learner_seat
        self.opp_seat = 1 - learner_seat
        self._next_seed = base_seed
        self._rng = random.Random(base_seed)

        sp = self._rpc({"cmd": "duel_spaces"})
        self.spaces_info = sp  # full handshake: shapes + env config (for params)
        self.n_actions = int(sp["n_actions"])
        self.obs_len = int(sp["obs_len"])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_len,), dtype=np.float32)
        self._mask = np.ones(self.n_actions, dtype=bool)
        # Every opponent keeps its resolver binding, so live sources can refresh only at a reset boundary.
        resolver = ControllerResolver()
        self.opp_pool = bind_opponents(opponent_models, resolver)
        self._validate_opponent_geometry()
        self.opp = self.opp_pool[0]

    def _rpc(self, msg: dict) -> dict:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed unexpectedly")
        return json.loads(line)

    def _scripted(self):
        """True if the current opponent is a server-side scripted agent ('greedy'/'random'), not a model."""
        return self.opp.resolved.model is None

    def _pick_opponent(self):
        """Sample this episode's opponent. A scripted anchor (e.g. greedy) is picked ~half the time when
        present — it's decisive, so it punishes passivity and keeps self-play from collapsing to draws."""
        scripted = [o for o in self.opp_pool if self._is_scripted_opponent(o)]
        models = [o for o in self.opp_pool if not self._is_scripted_opponent(o)]
        if scripted and (not models or self._rng.random() < 0.5):
            self.opp = self._rng.choice(scripted)
        else:
            self.opp = self._rng.choice(models)

    @staticmethod
    def _is_scripted_opponent(opponent):
        return opponent.resolved.model is None

    def _validate_opponent_geometry(self) -> None:
        for binding in self.opp_pool:
            resolved = binding.resolved
            if resolved.model is None:
                continue
            if resolved.observation_size != self.obs_len or resolved.action_size != self.n_actions:
                raise ControllerResolutionError("self-play opponent geometry does not match duel spaces")

    def _reload_live_opponents(self) -> None:
        """Refresh live sources only between episodes, before the next duel reset."""
        for binding in self.opp_pool:
            binding.reload()
        self._validate_opponent_geometry()

    def _play_opponent(self, v):
        """Drive a MODEL opponent (Python predict) until it's the learner's turn. Returns (view, reward).
        Scripted opponents are played server-side, so this is only used for model opponents."""
        acc = 0.0
        while not v["terminated"] and not v["truncated"] and int(v["seat"]) == self.opp_seat:
            observation = np.asarray(v["obs"], dtype=np.float32)
            mask = np.asarray(v["mask"], dtype=bool)
            resolved = self.opp.resolved
            assert resolved.model is not None and resolved.algorithm is not None
            validate_inference_input(resolved, observation, mask)
            a = predict_resolved(resolved.model, resolved.algorithm, observation, mask)
            v = self._rpc({"cmd": "duel_step", "action": int(a)})
            acc += float(v["reward"])
        return v, acc

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._next_seed
            self._next_seed += 1
        if not hasattr(self, "_rng"):
            self._rng = random.Random()
        self._rng.seed(int(seed))
        self._reload_live_opponents()
        self._pick_opponent()
        msg = {"cmd": "duel_reset", "seed": int(seed), "learner": self.learner}
        if self._scripted():
            msg[f"p{self.opp_seat}"] = self.opp.resolved.server_controller
        v = self._rpc(msg)
        if not self._scripted():
            v, _ = self._play_opponent(v)  # model opponent: Python drives it (incl. if it moves first)
        self._mask = np.asarray(v["mask"], dtype=bool)
        return np.asarray(v["obs"], dtype=np.float32), {}

    def step(self, action):
        # one duel_step covers the learner's move; for a scripted opponent the server also plays its reply
        # within that step (reward already includes it), so we only Python-drive a model opponent.
        v = self._rpc({"cmd": "duel_step", "action": int(action)})
        reward = float(v["reward"])
        if not self._scripted():
            v, acc = self._play_opponent(v)
            reward += acc
        self._mask = np.asarray(v["mask"], dtype=bool)
        return (np.asarray(v["obs"], dtype=np.float32), reward,
                bool(v["terminated"]), bool(v["truncated"]), {})

    def action_masks(self):
        return self._mask

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
