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
# Normalizer (Must match the logic used in training)
# ============================================================
class Normalizer:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)
        self.device = device

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.mean = data['mean']
        self.std = data['std']

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return (x * self.std) + self.mean

# ============================================================
# Deterministic World Model Wrapper
# ============================================================
class LearnedBicopterDynamics(nn.Module):
    def __init__(self, model, state_norm, action_norm, device="cpu"):
        super().__init__()
        self.model = model
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.device = device

    def reset(self):
        # For a standard MLP world model, no hidden state reset is needed
        pass

    def step(self, state, action):
        # 1. Normalize Inputs
        norm_s = self.state_norm.encode(state)
        norm_a = self.action_norm.encode(action)
        
        inp = torch.cat([norm_s, norm_a], dim=1)
        
        # 2. Predict (Model outputs normalized next state)
        norm_next_s = self.model(inp)
        
        # 3. Decode to real world units
        next_state = self.state_norm.decode(norm_next_s)
        
        return next_state

# ============================================================
# Rollout Function
# ============================================================
def rollout_policy(state0, policy, dynamics, traj_gen, control_mode, steps, dt=0.01, is_world_model=False, real_actions=None):
    states = state0.clone().unsqueeze(0) # (1, 6)
    trajectory = []
    targets = []
    actions_list = []

    if is_world_model:
        dynamics.reset()

    for t in range(steps):
        # Extract individual state components for observation
        # Note: Indexing depends on your state_dim (x, y, vx, vy, theta, omega)
        s_flat = states.squeeze(0)
        x, y, vx, vy, theta, omega = s_flat[0], s_flat[1], s_flat[2], s_flat[3], s_flat[4], s_flat[5]
        
        pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)
        pos_ref, vel_ref, acc_ref = pos_ref.squeeze(0), vel_ref.squeeze(0), acc_ref.squeeze(0)

        obs = torch.stack([
            pos_ref[0] - x, pos_ref[1] - y,
            vel_ref[0] - vx, vel_ref[1] - vy,
            acc_ref[0], acc_ref[1],
            torch.sin(theta), torch.cos(theta), omega
        ], dim=0).unsqueeze(0)

        if is_world_model and real_actions is not None:
            action = real_actions[t].unsqueeze(0)
        else:
            action = policy(obs)

        if is_world_model:
            states = dynamics.step(states, action)
        else:
            states = dynamics.step(states, action, control_mode=control_mode)

        trajectory.append(states.squeeze(0))
        targets.append(pos_ref)
        actions_list.append(action.squeeze(0))

    return torch.stack(trajectory), torch.stack(targets), torch.stack(actions_list)

# ============================================================
# Main
# ============================================================
@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def test(cfg: DictConfig):
    device = torch.device("cpu")
    steps = 200 # Longer rollout to see error compounding
    dt = 0.01
    cm = "srt"
    ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)

    # 1. Load Policy
    # Assuming MLP or BicopterPolicy is compatible
    policy = MLP(input=9, hidden=64, output=ACT_DIMS[cm]).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()

    # 2. Load World Model and Normalizers
    state_norm = Normalizer(6, device)
    action_norm = Normalizer(ACT_DIMS[cm], device)
    
    # Load the normalizer stats saved during training
    # (Note: You'll need to save these in your training script!)
    state_norm.load(os.path.join(output_dir, "state_norm.pt"))
    action_norm.load(os.path.join(output_dir, "action_norm.pt"))

    wm_net = MLP(input=6 + ACT_DIMS[cm], hidden=256, output=6).to(device)
    wm_net.load_state_dict(torch.load(os.path.join(output_dir, "world_model.pt"), map_location=device))
    wm_net.eval()

    learned_dynamics = LearnedBicopterDynamics(wm_net, state_norm, action_norm, device=device)

    # 3. Setup Real Dynamics and Traj Gen
    real_dynamics = BicopterDynamics(cfg=cfg, device=device)
    traj_gen = RandomTrajectoryGenerator(num_envs=1, device=device)

    state0 = torch.zeros(6, device=device)
    state0[0:2] = torch.tensor([1.0, 1.0]) # Start at a slight offset

    # 4. Rollouts
    with torch.inference_mode():
        real_traj, real_target, real_actions = rollout_policy(
            state0, policy, real_dynamics, traj_gen, cm, steps, dt, is_world_model=False
        )

        wm_traj, wm_target, wm_actions = rollout_policy(
            state0, policy, learned_dynamics, traj_gen, cm, steps, dt, is_world_model=True, real_actions=real_actions
        )

    # 5. Rendering
    renderer = MultiTrajectoryRenderer(drone=real_dynamics, video_path=None)
    renderer.add_agent(real_traj, real_target, real_actions, cm, (0, 0, 255), "REAL")
    renderer.add_agent(wm_traj, wm_target, real_actions, cm, (255, 0, 0), "WORLD_MODEL")

    print(f"Mean distance error: {torch.norm(real_traj[:, :2] - wm_traj[:, :2], dim=-1).mean():.4f}m")
    renderer.run()

if __name__ == "__main__":
    test()