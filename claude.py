import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rand_traj_gen import RandomTrajectoryGenerator
from utils.renderer import MultiTrajectoryRenderer
from dynamics.bicopter_dynamics import BicopterDynamics
from utils.nn import *
from utils.randomizer import env_randomization

import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================

STATE_DIM = 6
STATE_LABELS = ["x", "y", "vx", "vy", "theta", "omega"]
DT = 0.01


# ============================================================
# Normalizer
# ============================================================

class Normalizer:
    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)
        self.device = device

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-6

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def save(self, path: str):
        torch.save({'mean': self.mean, 'std': self.std}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.mean = data['mean']
        self.std = data['std']


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, act_dim), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.ptr = 0
        self.size = 0

    def add(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor):
        n = min(states.shape[0], self.capacity - self.ptr)
        self.states[self.ptr:self.ptr + n] = states[:n]
        self.actions[self.ptr:self.ptr + n] = actions[:n]
        self.next_states[self.ptr:self.ptr + n] = next_states[:n]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,))
        return self.states[idx], self.actions[idx], self.next_states[idx]


# ============================================================
# Observation Builder
# ============================================================

def build_obs(states: torch.Tensor, pos_ref: torch.Tensor, vel_ref: torch.Tensor, acc_ref: torch.Tensor) -> torch.Tensor:
    """Constructs the policy observation from current state and reference trajectory."""
    x, y, vx, vy, theta, omega = states.unbind(dim=-1)
    return torch.stack([
        pos_ref[:, 0] - x,
        pos_ref[:, 1] - y,
        vel_ref[:, 0] - vx,
        vel_ref[:, 1] - vy,
        acc_ref[:, 0],
        acc_ref[:, 1],
        torch.sin(theta),
        torch.cos(theta),
        omega,
    ], dim=1)


# ============================================================
# Rollout
# ============================================================

def collect_rollout(drone, policy, traj_gen, num_envs: int, steps: int,
                    control_mode: str, device: torch.device, store: bool = True):
    """Runs a policy rollout on the real dynamics. Returns (states, actions) tensors."""
    states = torch.zeros((num_envs, STATE_DIM), device=device)
    all_states, all_actions = [], []

    with torch.no_grad():
        for step in range(steps):
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(step * DT)
            obs = build_obs(states, pos_ref, vel_ref, acc_ref)
            actions = policy(obs)
            next_states = drone.step(states, actions, control_mode=control_mode)

            if store:
                all_states.append(states)
                all_actions.append(actions)

            states = next_states

    if store:
        return torch.cat(all_states), torch.cat(all_actions)
    return None, None


# ============================================================
# Training
# ============================================================

def train(control_mode: str, cfg: DictConfig, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    act_dim = ACT_DIMS[control_mode]
    num_envs = 2048

    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    policy = MLP(input=9, hidden=64, output=act_dim).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()

    # ── 1. Collect data and fit normalizers ──────────────────
    print("Collecting rollout data for normalization and replay buffer...")
    all_states, all_actions = collect_rollout(
        drone, policy, traj_gen, num_envs=num_envs,
        steps=500, control_mode=control_mode, device=device
    )

    state_norm = Normalizer(STATE_DIM, device)
    action_norm = Normalizer(act_dim, device)
    state_norm.fit(all_states)
    action_norm.fit(all_actions)

    # ── 2. Fill replay buffer ─────────────────────────────────
    buffer = ReplayBuffer(capacity=1_000_000, state_dim=STATE_DIM, act_dim=act_dim, device=device)

    # Reconstruct next_states by pairing consecutive states
    # (state[t], action[t]) → next_state[t+1] per env
    # Since collect_rollout stores states before stepping, we re-run a short
    # rollout to get (s, a, s') triples cleanly.
    states = torch.zeros((num_envs, STATE_DIM), device=device)
    with torch.no_grad():
        for step in range(500):
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(step * DT)
            obs = build_obs(states, pos_ref, vel_ref, acc_ref)
            actions = policy(obs)
            next_states = drone.step(states, actions, control_mode=control_mode)
            buffer.add(states, actions, next_states)
            states = next_states

    print(f"Replay buffer size: {buffer.size}")

    # ── 3. Build and train world model ────────────────────────
    world_model = MLP(input=STATE_DIM + act_dim, hidden=256, output=STATE_DIM).to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    epochs = 200
    batch_size = 4096
    steps_per_epoch = 800

    print("Training world model...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            s, a, s_next = buffer.sample(batch_size)
            wm_input = torch.cat([state_norm.encode(s), action_norm.encode(a)], dim=-1)
            pred_next_norm = world_model(wm_input)
            loss = F.mse_loss(pred_next_norm, state_norm.encode(s_next))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {epoch_loss / steps_per_epoch:.6f}")

    # ── 4. Save ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    torch.save(world_model.state_dict(), os.path.join(output_dir, "world_model.pt"))
    state_norm.save(os.path.join(output_dir, "state_norm.pt"))
    action_norm.save(os.path.join(output_dir, "action_norm.pt"))
    print("World model and normalizers saved.")

    return world_model, state_norm, action_norm


# ============================================================
# Evaluation
# ============================================================

def evaluate_rollout(world_model, drone, traj_gen, state_norm: Normalizer,
                     action_norm: Normalizer, policy, control_mode: str, device: torch.device):
    """Compares real dynamics vs. world model over a single rollout."""
    world_model.eval()
    policy.eval()
    rollout_steps = 200

    state_real = torch.zeros((1, STATE_DIM), device=device)
    state_model = state_real.clone()
    real_traj, model_traj, action_traj, target_traj = [], [], [], []

    with torch.no_grad():
        for t in range(rollout_steps):
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * DT)
            obs = build_obs(state_real, pos_ref, vel_ref, acc_ref)
            action = policy(obs)

            # Real dynamics step
            next_real = drone.step(state_real, action, control_mode=control_mode)

            # World model step
            wm_input = torch.cat([state_norm.encode(state_model), action_norm.encode(action)], dim=-1)
            pred_next = state_norm.decode(world_model(wm_input))

            real_traj.append(next_real.cpu())
            model_traj.append(pred_next.cpu())
            action_traj.append(action.cpu())
            target_traj.append(pos_ref.squeeze(0).cpu())

            state_real = next_real
            state_model = pred_next

    return (
        torch.cat(real_traj),
        torch.cat(model_traj),
        torch.cat(action_traj),
        target_traj,
    )


def plot_rollout_comparison(real_traj: torch.Tensor, model_traj: torch.Tensor):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (ax, label) in enumerate(zip(axes.flat, STATE_LABELS)):
        ax.plot(real_traj[:, i], label="real", linewidth=2)
        ax.plot(model_traj[:, i], label="model", linewidth=2)
        ax.set_title(f"Rollout comparison: {label}")
        ax.set_xlabel("timestep")
        ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def main(cfg: DictConfig):
    control_mode = "srt"
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", control_mode)

    # ── Train ─────────────────────────────────────────────────
    start = time.time()
    world_model, state_norm, action_norm = train(control_mode, cfg, output_dir)
    print(f"Training time: {time.time() - start:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────
    device = torch.device("cpu")
    act_dim = ACT_DIMS[control_mode]

    drone = BicopterDynamics(cfg=cfg, device=device)
    traj_gen = RandomTrajectoryGenerator(num_envs=1, device=device)

    policy = MLP(input=9, hidden=64, output=act_dim).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()

    eval_state_norm = Normalizer(STATE_DIM, device)
    eval_action_norm = Normalizer(act_dim, device)
    eval_state_norm.load(os.path.join(output_dir, "state_norm.pt"))
    eval_action_norm.load(os.path.join(output_dir, "action_norm.pt"))

    eval_world_model = MLP(input=STATE_DIM + act_dim, hidden=256, output=STATE_DIM).to(device)
    eval_world_model.load_state_dict(torch.load(os.path.join(output_dir, "world_model.pt"), map_location=device))

    real_traj, model_traj, action_traj, target_traj = evaluate_rollout(
        eval_world_model, drone, traj_gen,
        eval_state_norm, eval_action_norm,
        policy, control_mode, device
    )

    plot_rollout_comparison(real_traj, model_traj)
    print(f"Rollout MSE: {F.mse_loss(real_traj, model_traj):.6f}")

    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)
    renderer.add_agent(real_traj, target_traj, action_traj, control_mode, (0, 0, 255), "REAL")
    renderer.add_agent(model_traj, target_traj, action_traj, control_mode, (255, 0, 0), "WORLD_MODEL")
    renderer.run()


if __name__ == "__main__":
    ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}
    main()