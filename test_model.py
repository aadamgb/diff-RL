import torch
from utils.renderer import MultiTrajectoryRenderer
from utils.nn import *
from utils.rand_traj_gen import RandomTrajectoryGenerator
from utils.randomizer import env_randomization
from dynamics.bicopter_dynamics import BicopterDynamics
import os

import hydra
from omegaconf import DictConfig

from collections import deque
import matplotlib.pyplot as plt

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def test(cfg: DictConfig):
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # One Rollout
    # -----------------------------------------------------------------------------
    def rollout_policy(
        state0,
        # z_true,
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
        # z_hat_history = []

        states = state0
        
        def get_target_safe(time):
            pos, vel, acc = traj_gen.get_target(max(time, 0.0))
            return pos.squeeze(0), vel.squeeze(0), acc.squeeze(0)
        
        history = deque(maxlen=20)   # fixed window
        z_hat = torch.zeros(2)

        for t in range(steps):
            x, y, vx, vy, theta, omega = states.squeeze(0)
            # Get reference trajectory
            pos_ref, vel_ref, acc_ref = get_target_safe(t * dt)

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

            # if len(history) == 20:
            #     hist_tensor = torch.stack(list(history), dim=0)
            #     z_hat = adapt_module(hist_tensor.unsqueeze(0)).squeeze(0)

            # z_hat_history.append(z_hat.clone())

            # obs = torch.cat([obs, z], dim=0)  # added ecoded env_params
            # obs = torch.cat([obs, z_hat], dim=0)  # added ecoded env_params

            # Update state
            actions = policy(obs)
            history.append(torch.cat([states, actions], dim=0))
            states = drone.step(states, actions, control_mode=control_mode).squeeze()

            # rand_mass = {"m" : 0.2}
            # e_new =  env_randomization(cfg, num_envs=1)
            # if t == 800:
            #     drone.randomize_parameters(e_new)
            #     print(f"New newparams{e_new}")

            eval_traj.append(states)
            eval_target.append(pos_ref)
            eval_actions.append(actions)

        return (
            torch.stack(eval_traj),
            torch.stack(eval_target),
            torch.stack(eval_actions),
            # torch.stack(z_hat_history),
        )

    # -----------------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------------
    steps = 1500 * 3
    dt = 0.01
    num_envs = 1
    device = "cpu"

    drone = BicopterDynamics(cfg=cfg)
    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)
    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)
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
            state0 = torch.zeros(6)
            env_params = env_randomization(cfg, num_envs=1)
            # print(f"Initial params {env_params}")
            drone.randomize_parameters(env_params)
            # e = torch.stack([env_params["m"], env_params["J"], env_params["l"], env_params["C_Dx"], env_params["C_Dy"]], dim=1)
            policy = BicopterPolicy(
                obs_dim=9, 
                act_dim=ACT_DIMS[cm]
            )
            adapt_module = AdaptationModule(input_dim=(6 + ACT_DIMS[cm]))

            policy_path = os.path.join(output_dir, cm, "policy.pt")
            # encoder_path = os.path.join(output_dir, cm, "encoder.pt")
            # adapt_path = os.path.join(output_dir, cm, "adapt_module.pt")
            # if not os.path.exists(policy_path) or not os.path.exists(encoder_path):
            #     print(f"Warning: Model file not found. Skipping {cm.upper()}.")
            #     continue

            policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
            # env_encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
            # adapt_module.load_state_dict(torch.load(adapt_path, map_location="cpu"))

            policy.eval()
            # env_encoder.eval()
            # adapt_module.eval()


            # z_true = env_encoder(e).detach()

            eval_traj, eval_target, eval_actions = rollout_policy(
                state0=state0,
                # z_true=z_true,
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
                # z_hat_history=z_hat_history,
                # z_true=z_true,
            )

            print(f"Loaded and rendered {cm.upper()} policy")

    renderer.run()
    # renderer.plot_dashboard()

if __name__ == "__main__":
    test()