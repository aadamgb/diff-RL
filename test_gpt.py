import torch
import torch.nn as nn
import os
import hydra
from omegaconf import DictConfig

from utils.renderer import MultiTrajectoryRenderer
from utils.nn import *
from utils.rand_traj_gen import RandomTrajectoryGenerator
from utils.randomizer import env_randomization
from dynamics.bicopter_dynamics import BicopterDynamics


# ============================================================
# Deterministic World Model Wrapper
# ============================================================

class LearnedBicopterDynamics(nn.Module):
    def __init__(self, obs_encoder, sequence_model, obs_decoder, act_dim, device="cpu"):
        super().__init__()
        self.encoder = obs_encoder
        self.gru = sequence_model
        self.decoder = obs_decoder
        self.device = device
        self.hidden_size = sequence_model.hidden_size
        self.act_dim = act_dim
        self.reset()

    def reset(self):
        self.h = torch.zeros(1, 1, self.hidden_size, device=self.device)

    def step(self, state, action):
        """
        state: (1, 6)
        action: (1, act_dim)
        """
        e = self.encoder(state)                     # (1, 32)
        inp = torch.cat([e, action], dim=1)         # (1, 32 + act_dim)
        inp = inp.unsqueeze(1)                      # (1, 1, features)

        out, self.h = self.gru(inp, self.h)         # (1,1,64)
        next_state = self.decoder(out.squeeze(1))   # (1,6)

        return next_state


# ============================================================
# Rollout Function
# ============================================================

def rollout_policy(
    state0,
    policy,
    dynamics,
    traj_gen,
    control_mode,
    steps,
    dt=0.01,
    is_world_model=False
):
    states = state0.clone().unsqueeze(0)
    trajectory = []
    targets = []
    actions_list = []

    if is_world_model:
        dynamics.reset()

    for t in range(steps):

        x, y, vx, vy, theta, omega = states.squeeze(0)
        pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)
        pos_ref = pos_ref.squeeze(0)
        vel_ref = vel_ref.squeeze(0)
        acc_ref = acc_ref.squeeze(0)

        obs = torch.stack([
            pos_ref[0] - x,
            pos_ref[1] - y,
            vel_ref[0] - vx,
            vel_ref[1] - vy,
            acc_ref[0],
            acc_ref[1],
            torch.sin(theta),
            torch.cos(theta),
            omega
        ], dim=0).unsqueeze(0)

        action = policy(obs)

        if is_world_model:
            states = dynamics.step(states, action)
        else:
            states = dynamics.step(states, action, control_mode=control_mode)

        trajectory.append(states.squeeze(0))
        targets.append(pos_ref)
        actions_list.append(action.squeeze(0))

    return (
        torch.stack(trajectory),
        torch.stack(targets),
        torch.stack(actions_list),
    )


# ============================================================
# Main
# ============================================================

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def test(cfg: DictConfig):

    device = torch.device("cpu")
    steps = 3000
    dt = 0.01
    cm = "srt"

    ACT_DIMS = {
        "ctbr": 2,
        "srt": 2,
        "lv": 5
    }

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)

    # --------------------------------------------------------
    # Load Policy
    # --------------------------------------------------------

    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cm]).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()

    # --------------------------------------------------------
    # Load World Model
    # --------------------------------------------------------

    obs_encoder = MLP(input=6, hidden=64, output=12).to(device)
    sequence_model = nn.GRU(
        input_size=12 + ACT_DIMS[cm],
        hidden_size=64,
        batch_first=True
    ).to(device)
    obs_decoder = MLP(input=64, hidden=64, output=6).to(device)

    obs_encoder.load_state_dict(torch.load(os.path.join(output_dir, "obs_encoder.pt"), map_location=device))
    sequence_model.load_state_dict(torch.load(os.path.join(output_dir, "sequence_model.pt"), map_location=device))
    obs_decoder.load_state_dict(torch.load(os.path.join(output_dir, "obs_decoder.pt"), map_location=device))

    obs_encoder.eval()
    sequence_model.eval()
    obs_decoder.eval()

    learned_dynamics = LearnedBicopterDynamics(
        obs_encoder,
        sequence_model,
        obs_decoder,
        ACT_DIMS[cm],
        device=device
    )

    # --------------------------------------------------------
    # Real Dynamics
    # --------------------------------------------------------

    real_dynamics = BicopterDynamics(cfg=cfg)
    env_params = env_randomization(cfg, num_envs=1)
    real_dynamics.randomize_parameters(env_params)

    # --------------------------------------------------------
    # Trajectory Generator
    # --------------------------------------------------------

    traj_gen = RandomTrajectoryGenerator(num_envs=1, device=device)

    state0 = torch.zeros(6)
    state0[0] = -3.0
    state0[1] = 3.0

    # --------------------------------------------------------
    # Rollouts
    # --------------------------------------------------------

    with torch.inference_mode():

        real_traj, real_target, real_actions = rollout_policy(
            state0,
            policy,
            real_dynamics,
            traj_gen,
            cm,
            steps,
            dt,
            is_world_model=False
        )

        wm_traj, wm_target, wm_actions = rollout_policy(
            state0,
            policy,
            learned_dynamics,
            traj_gen,
            cm,
            steps,
            dt,
            is_world_model=True
        )

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    renderer = MultiTrajectoryRenderer(drone=real_dynamics, video_path=None)

    renderer.add_agent(
        trajectory=real_traj,
        target_trajectory=real_target,
        action=real_actions,
        control_mode=cm,
        color=(0, 0, 255),
        name="REAL"
    )

    renderer.add_agent(
        trajectory=wm_traj,
        target_trajectory=wm_target,
        action=wm_actions,
        control_mode=cm,
        color=(255, 0, 0),
        name="WORLD_MODEL"
    )

    print("Rendering Real vs World Model")
    renderer.run()


if __name__ == "__main__":
    test()
