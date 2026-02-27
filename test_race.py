import torch
from utils.renderer import MultiTrajectoryRenderer
from utils.nn import *
from env.race_env import RaceEnv
from dynamics.bicopter_dynamics import BicopterDynamics
import os

import hydra
from omegaconf import DictConfig

import numpy as np

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def test(cfg: DictConfig):
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # One Rollout using RaceEnv
    # -------------------------------------------------------------------------
    def rollout_policy(env, policy, control_mode, steps):
        eval_traj = []
        eval_target = []
        eval_actions = []

        obs = env.reset()  # obs shape: (num_envs, obs_dim)

        for t in range(steps):
            # Policy inference
            actions = policy(obs)

            # Environment step
            obs = env.step(actions)

            # Track trajectory: use environment states
            eval_traj.append(env.states.clone())
            eval_target.append(env.pos_ref.clone())
            eval_actions.append(actions.clone() if hasattr(actions, 'clone') else actions)

        return (
            torch.cat(eval_traj, dim=0),
            torch.cat(eval_target, dim=0),
            torch.cat(eval_actions, dim=0),
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    steps = 1500
    num_envs = 1
    device = "cpu"

    drone = BicopterDynamics(cfg=cfg)
    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)

    # Create RaceEnv and pass track gates to renderer
    env = RaceEnv(cfg, num_envs, device)
    gates = [
        {"position": env.gate_positions[i].tolist(), "look_at": env.gate_orientations[i].tolist()}
        for i in range(env.num_gates)
    ]
    renderer.set_track(gates)

    ACT_DIMS = {
        "srt": 2,       # T1, T2
        "ctbr": 2,      # T, tau
        "lv": 5         # vx, vy, kv, kR, kw
    }

    control_modes = {
        "srt": {"color": (0, 255, 0)},
        "ctbr": {"color": (0, 0, 255)},
        "lv": {"color": (255, 165, 0)},
    }

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    with torch.inference_mode():
        for cm, config in control_modes.items():
            policy = BicopterPolicy(
                obs_dim=9,
                act_dim=ACT_DIMS[cm]
            )

            policy_path = os.path.join(output_dir, cm, "race_policy.pt")

            if not os.path.exists(policy_path):
                print(f"Warning: Model file not found. Skipping {cm.upper()}.")
                continue

            policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
            policy.eval()

            # Run rollout with this control mode
            eval_traj, eval_target, eval_actions = rollout_policy(
                env=env,
                policy=policy,
                control_mode=cm,
                steps=steps,
            )

            # Convert to proper shapes for renderer (squeeze num_envs dimension)
            renderer.add_agent(
                trajectory=eval_traj.squeeze(1),  # (steps, 6)
                target_trajectory=eval_target.squeeze(1),  # (steps, 2)
                action=eval_actions.squeeze(1),  # (steps, act_dim)
                control_mode=cm,
                color=config["color"],
                name=cm.upper(),
            )

            print(f"Loaded and rendered {cm.upper()} policy")

    renderer.run()

if __name__ == "__main__":
    test()
