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
# Trajectory Buffer
#
# Stores full rollout trajectories so we can sample contiguous
# sequences of length k for multi-step rollout training.
#
# Layout: episodes are stored as (T, num_envs, dim) slices.
# Sampling picks a random env and a random start timestep,
# then returns a window of length k.
# ============================================================

class TrajectoryBuffer:
    def __init__(self, max_steps: int, num_envs: int, state_dim: int,
                 act_dim: int, device: torch.device):
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.device = device

        # Pre-allocate: (max_steps, num_envs, dim)
        self.states  = torch.zeros((max_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((max_steps, num_envs, act_dim),   device=device)

        self.t = 0          # next write position (timestep axis)
        self.full = False   # whether the buffer has wrapped around

    def add_step(self, states: torch.Tensor, actions: torch.Tensor):
        """Store one timestep across all envs. Call once per env-step."""
        self.states[self.t]  = states
        self.actions[self.t] = actions
        self.t = (self.t + 1) % self.max_steps
        if self.t == 0:
            self.full = True

    @property
    def filled_steps(self) -> int:
        return self.max_steps if self.full else self.t

    def sample_sequences(self, batch_size: int, horizon: int):
        """
        Returns (s0, actions_seq, states_seq) where:
          s0          : (batch, state_dim)   – starting state
          actions_seq : (batch, horizon, act_dim)
          states_seq  : (batch, horizon, state_dim)  – ground-truth next states

        A valid window [t, t+horizon] must not straddle the write pointer
        (which would mix old and new data), so we exclude those windows.
        """
        T = self.filled_steps
        assert T > horizon, "Not enough data to sample sequences of this length."

        # Sample random (env, start_t) pairs
        env_idx   = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        # Exclude the last `horizon` steps before the write pointer to avoid wrap-around artifacts
        valid_T   = T - horizon
        start_idx = torch.randint(0, valid_T, (batch_size,), device=self.device)

        # Build index tensor: (batch, horizon)
        offsets = torch.arange(horizon, device=self.device).unsqueeze(0)  # (1, horizon)
        t_idx   = (start_idx.unsqueeze(1) + offsets) % self.max_steps      # (batch, horizon)

        # Gather: states_seq[b, h] = states[t_idx[b,h], env_idx[b]]
        states_seq  = self.states[t_idx, env_idx.unsqueeze(1)]   # (batch, horizon, state_dim)
        actions_seq = self.actions[t_idx, env_idx.unsqueeze(1)]  # (batch, horizon, act_dim)

        s0 = states_seq[:, 0, :]                   # (batch, state_dim)
        # next-state targets: states at t+1 through t+horizon
        next_t_idx     = (t_idx + 1) % self.max_steps
        next_states_seq = self.states[next_t_idx, env_idx.unsqueeze(1)]  # (batch, horizon, state_dim)

        return s0, actions_seq, next_states_seq


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
# Training
# ============================================================

def multistep_rollout_loss(
    world_model: nn.Module,
    s0: torch.Tensor,
    actions_seq: torch.Tensor,
    targets_seq: torch.Tensor,
    state_norm: Normalizer,
    action_norm: Normalizer,
) -> torch.Tensor:
    """
    Unrolls the world model for `horizon` steps starting from s0,
    feeding its own predictions back as inputs at each step.

    Args:
        s0          : (batch, state_dim)          – initial state (un-normalised)
        actions_seq : (batch, horizon, act_dim)   – action sequence (un-normalised)
        targets_seq : (batch, horizon, state_dim) – ground-truth next states (un-normalised)

    Returns:
        Scalar MSE loss averaged over all steps and batch elements.
        Later steps are weighted more heavily to prioritise long-horizon accuracy.
    """
    horizon = actions_seq.shape[1]
    s_pred = s0
    loss = torch.tensor(0.0, device=s0.device)

    for t in range(horizon):
        a_t = actions_seq[:, t, :]
        wm_input = torch.cat([state_norm.encode(s_pred), action_norm.encode(a_t)], dim=-1)
        s_pred_norm = world_model(wm_input)
        s_pred = state_norm.decode(s_pred_norm)  # back to raw space for the next step

        target_norm = state_norm.encode(targets_seq[:, t, :])

        # Linear ramp: weight step t from 1.0 to 2.0 so later steps matter more
        step_weight = 1.0 + t / max(horizon - 1, 1)
        loss = loss + step_weight * F.mse_loss(s_pred_norm, target_norm)

    return loss / horizon


def train(control_mode: str, cfg: DictConfig, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    act_dim  = ACT_DIMS[control_mode]
    num_envs = 2048

    # ── Hyperparameters ───────────────────────────────────────
    rollout_steps   = 5000
    epochs          = 200
    batch_size      = 1024   # smaller than before since we unroll k steps
    steps_per_epoch = 1000
    # max_horizon     = 10     # final unroll length
    max_horizon     = 100     # final unroll length

    # Curriculum: ramp horizon from 1 → max_horizon over the first half of training
    horizon_warmup_epochs = epochs // 2

    drone    = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    policy = MLP(input=9, hidden=64, output=act_dim).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()

    # ── 1. Collect trajectories ───────────────────────────────
    print("Collecting rollout data...")
    buffer = TrajectoryBuffer(
        max_steps=rollout_steps,
        num_envs=num_envs,
        state_dim=STATE_DIM,
        act_dim=act_dim,
        device=device,
    )

    states = torch.zeros((num_envs, STATE_DIM), device=device)
    all_states_list, all_actions_list = [], []

    with torch.no_grad():
        for step in range(rollout_steps):
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(step * DT)
            obs     = build_obs(states, pos_ref, vel_ref, acc_ref)
            actions = policy(obs)
            next_states = drone.step(states, actions, control_mode=control_mode)

            buffer.add_step(states, actions)
            all_states_list.append(states)
            all_actions_list.append(actions)

            states = next_states

    print(f"Trajectory buffer filled: {buffer.filled_steps} steps × {num_envs} envs")

    # ── 2. Fit normalizers ────────────────────────────────────
    all_states  = torch.cat(all_states_list)
    all_actions = torch.cat(all_actions_list)

    state_norm  = Normalizer(STATE_DIM, device)
    action_norm = Normalizer(act_dim, device)
    state_norm.fit(all_states)
    action_norm.fit(all_actions)

    # ── 3. Build world model ──────────────────────────────────
    world_model = MLP(input=STATE_DIM + act_dim, hidden=256, output=STATE_DIM).to(device)
    optimizer   = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    # ── 4. Train with multi-step rollout loss + curriculum ────
    print(f"Training world model (horizon curriculum: 1 → {max_horizon} over {horizon_warmup_epochs} epochs)...")

    for epoch in range(epochs):

        # Curriculum: linearly increase horizon from 1 to max_horizon
        progress = min(epoch / horizon_warmup_epochs, 1.0)
        horizon  = max(1, round(1 + progress * (max_horizon - 1)))

        world_model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            s0, actions_seq, targets_seq = buffer.sample_sequences(batch_size, horizon)

            loss = multistep_rollout_loss(
                world_model, s0, actions_seq, targets_seq,
                state_norm, action_norm
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | horizon={horizon:2d} | Loss: {epoch_loss / steps_per_epoch:.6f}")

    # ── 5. Save ───────────────────────────────────────────────
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
    # world_model, state_norm, action_norm = train(control_mode, cfg, output_dir)
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