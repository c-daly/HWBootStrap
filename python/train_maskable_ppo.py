"""Smoke-train a HexWars tactical policy with MaskablePPO against the GreedyAgent.

Run inside WSL2 after building the server (see python/README.md):
    export HEXWARS_SERVER=$(pwd)/engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll
    cd python && python train_maskable_ppo.py
"""
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

from hexwars_gym import HexWarsEnv

DLL = os.environ.get(
    "HEXWARS_SERVER",
    "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll",
)
SERVER_CMD = ["dotnet", DLL]


def mask_fn(env) -> "list":
    return env.action_masks()


def make_env():
    env = HexWarsEnv(SERVER_CMD, opponent="greedy", seat=0)
    return ActionMasker(env, mask_fn)


def main():
    env = make_env()
    model = MaskablePPO(MaskableActorCriticPolicy, env, n_steps=512, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("hexwars_maskable_ppo")
    print("done -> hexwars_maskable_ppo.zip")
    env.close()


if __name__ == "__main__":
    main()
