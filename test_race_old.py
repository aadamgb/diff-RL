import torch
from utils.renderer import MultiTrajectoryRenderer
from utils.nn import *
from utils.rand_traj_gen import RandomTrajectoryGenerator
from utils.randomizer import env_randomization
from dynamics.bicopter_dynamics import BicopterDynamics
import os
import yaml

import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

from collections import deque
import matplotlib.pyplot as plt

import numpy as np

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def test(cfg: DictConfig):
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    def load_track_config():
        track_path = os.path.join(get_original_cwd(), "cfg", "track", f"{cfg.track.name}.yaml")
        with open(track_path, "r") as f:
            track_config = yaml.safe_load(f)
        return track_config.get("gates", [])

    # -----------------------------------------------------------------------------
    # One Rollout
    # -----------------------------------------------------------------------------
    def rollout_policy(
        state0,
        policy,
        drone,
        traj_gen,
        control_mode,
        steps,
        dt=0.01,
    ):
        eval_traj = []
        eval_target = []
        eval_actions = []

        states = state0
        timer = 0.0
        def get_target(boundary):
            pos, vel, acc = traj_gen.get_hover_targets(boundary)
            return pos.squeeze(0), vel.squeeze(0), acc.squeeze(0)
        
        pos_ref, vel_ref, acc_ref = get_target(boundary=5.0)
        
        for t in range(steps):
            distance = torch.sqrt(((pos_ref - states[:2])**2).sum())
            
            if distance < 1.0:
                timer += dt
            else:
                timer = 0.0

            if timer > 1.0:
            # Generate new target
                pos_ref, vel_ref, acc_ref = get_target(boundary=5.0)
                timer = 0.0

            x, y, vx, vy, theta, omega = states.squeeze(0)

            # Compute errors
            e_px = pos_ref[0] - x
            e_py = pos_ref[1] - y
            e_vx = vel_ref[0] - vx
            e_vy = vel_ref[1] - vy

            # Build observation input
            obs = torch.stack([
                e_px, e_py, e_vx, e_vy,
                acc_ref[0],
                acc_ref[1],
                torch.sin(theta),
                torch.cos(theta),
                omega
            ], dim=0)

            # Update state
            actions = policy(obs)
            states = drone.step(states, actions, control_mode=control_mode).squeeze()

            eval_traj.append(states)
            eval_target.append(pos_ref)
            eval_actions.append(actions)

        return (
            torch.stack(eval_traj),
            torch.stack(eval_target),
            torch.stack(eval_actions),
        )

    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------
    steps = 1500
    dt = 0.01
    num_envs = 1
    device = "cpu"

    drone = BicopterDynamics(cfg=cfg)
    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)
    renderer.set_track(load_track_config())
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    ACT_DIMS = {
        "srt": 2,       # T1, T2
        "ctbr": 2,      # T, tau
        "lv": 5         #  vx, vy, kv, kR, kw  
    }

    control_modes = {
        "srt": {"color": (0, 255, 0)},
        "ctbr": {"color": (0, 0, 255)},
        "lv": {"color": (255, 165, 0)},
    }

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    with torch.inference_mode():
        for cm, config in control_modes.items():
            gates = load_track_config()
            if gates:
                positions = np.array([gate['position'] for gate in gates])
                print(positions[0][0])
                x0, y0 = positions[0][0], positions[0][1]
            else:
                x0, y0 = 0.0, 0.0
            state0 = torch.tensor([x0, y0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

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


            eval_traj, eval_target, eval_actions = rollout_policy(
                state0=state0,
                policy=policy,
                drone=drone,
                traj_gen=traj_gen,
                control_mode=cm,
                steps=steps,
                dt=dt,
            )

            renderer.add_agent(
                trajectory=eval_traj,
                target_trajectory=eval_target,
                action=eval_actions,
                control_mode=cm,
                color=config["color"],
                name=cm.upper(),
            )

            print(f"Loaded and rendered {cm.upper()} policy")

    renderer.run()
    # renderer.plot_dashboard()

if __name__ == "__main__":
    test()