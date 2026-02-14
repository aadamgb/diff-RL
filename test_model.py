import torch
from utils.renderer import MultiTrajectoryRenderer
from utils.nn import BicopterPolicy
from utils.rand_traj_gen import RandomTrajectoryGenerator
from utils.randomizer import env_randomization
from dynamics.bicopter_dynamics import BicopterDynamics
import os

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def test(cfg: DictConfig):
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

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
        
        def get_target_safe(time):
            pos, vel, acc = traj_gen.get_target(max(time, 0.0))
            return pos.squeeze(0), vel.squeeze(0), acc.squeeze(0)  # (2,)

        for t in range(steps):
            x, y, vx, vy, theta, omega, Omega1, Omega2 = states.squeeze(0)

            # Get reference trajectory
            pos_ref, vel_ref, acc_ref = get_target_safe(t * dt)
            # pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)

            # Compute errors
            e_px = pos_ref[0] - x
            e_py = pos_ref[1] - y
            e_vx = vel_ref[0] - vx
            e_vy = vel_ref[1] - vy

            # Build observation
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
            states = drone.step(states, actions, control_mode=control_mode)

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
    steps = 1500 * 3
    dt = 0.01
    num_envs = 1
    device = "cpu"

    drone = BicopterDynamics(cfg=cfg)
    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    ACT_DIMS = {
        "srt": 2,       # T1, T2
        "ctbr": 2,      # T, tau
        "lv": 5         #  vx, vy, kv, kR, kw  
    }

    control_modes = {
        "srt": {"color": (0, 255, 0), "file_name": "srt.pt"},
        "ctbr": {"color": (0, 0, 255), "file_name": "ctbr.pt"},
        # "lv": {"color": (255, 165, 0), "file_name": "lv.pt"},
        "lv": {"color": (255, 165, 0), "file_name": "lv2.pt"},
    }

    # -----------------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------------
    with torch.inference_mode():
        state0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state0[6] = drone.motor_hover_speed()                       
        state0[7] = drone.motor_hover_speed()
        drone.randomize_parameters(env_randomization(cfg))

        for cm, config in control_modes.items():
            policy = BicopterPolicy(
                obs_dim=9, 
                act_dim=ACT_DIMS[cm]
            )

            policy.eval()

            model_path = os.path.join(output_dir, config["file_name"])
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found. Skipping {cm.upper()}.")
                continue

            policy.load_state_dict(torch.load(model_path, map_location="cpu"))

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