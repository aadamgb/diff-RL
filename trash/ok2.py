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
# Normalizer
# ============================================================

class Normalizer:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)
        self.device = device

    def update(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-6

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.mean = data['mean']
        self.std = data['std']


    def encode(self, x):
        return (x - self.mean) / self.std


# ============================================================
# Simple Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim, act_dim, device):
        self.capacity = capacity
        self.device = device

        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, act_dim), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)

        self.ptr = 0
        self.size = 0

    def add(self, s, a, s_next):
        n = s.shape[0]

        if self.ptr + n > self.capacity:
            n = self.capacity - self.ptr

        self.states[self.ptr:self.ptr+n] = s[:n]
        self.actions[self.ptr:self.ptr+n] = a[:n]
        self.next_states[self.ptr:self.ptr+n] = s_next[:n]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.states[idx],
            self.actions[idx],
            self.next_states[idx]
        )


# ============================================================
# Training
# ============================================================

def train(cm, cfg: DictConfig):

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm + "_copy")
    print(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    state_dim = 6
    act_dim = ACT_DIMS[cm]

    drone = BicopterDynamics(device=device, cfg=cfg)
    policy = MLP(input=9, hidden=64, output=ACT_DIMS[cm]).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()


    # --------------------------------------------------------
    # 1. Collect Real Rollout Data for Normalization
    # --------------------------------------------------------

    print("Collecting data for normalization...")

    states = torch.zeros((2048, state_dim), device=device)
    all_states = []
    all_actions = []
    
    traj_gen = RandomTrajectoryGenerator(num_envs=2048, device=device)
    dt = 0.01

    with torch.no_grad():
        for step in range(200):
            # Extract state components
            x = states[:, 0]
            y = states[:, 1]
            vx = states[:, 2]
            vy = states[:, 3]
            theta = states[:, 4]
            omega = states[:, 5]
            
            # Get reference trajectory
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(step * dt)
            
            # Build observation for policy
            obs = torch.stack([
                pos_ref[:, 0] - x,
                pos_ref[:, 1] - y,
                vel_ref[:, 0] - vx,
                vel_ref[:, 1] - vy,
                acc_ref[:, 0],
                acc_ref[:, 1],
                torch.sin(theta),
                torch.cos(theta),
                omega
            ], dim=1)
            
            # Get actions from policy
            actions = policy(obs)
            # actions = torch.tensor([[5.0, 5.0]], device=device)
            next_states = drone.step(states, actions, control_mode=cm)

            all_states.append(states)
            all_actions.append(actions)

            states = next_states

    all_states = torch.cat(all_states, dim=0)
    all_actions = torch.cat(all_actions, dim=0)

    state_norm = Normalizer(state_dim, device)
    action_norm = Normalizer(act_dim, device)

    state_norm.update(all_states)
    action_norm.update(all_actions)

    print("Normalization computed from real data.")

    # --------------------------------------------------------
    # 2. World Model
    # --------------------------------------------------------

    # world_model = MLP(
    #     input=state_dim + act_dim,
    #     hidden=256,
    #     output=state_dim
    # ).to(device)

    state_norm.load(os.path.join(output_dir, "state_norm.pt"))
    action_norm.load(os.path.join(output_dir, "action_norm.pt"))

    world_model = MLP(input=6 + ACT_DIMS[cm], hidden=256, output=6).to(device)
    world_model.load_state_dict(torch.load(os.path.join(output_dir, "world_model.pt"), map_location=device))
    print(f"Loaded WM from: {output_dir}")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)

    # --------------------------------------------------------
    # 3. Replay Buffer
    # --------------------------------------------------------

    buffer = ReplayBuffer(
        capacity=1_000_000,
        state_dim=state_dim,
        act_dim=act_dim,
        device=device
    )

    # --------------------------------------------------------
    # 4. Collect Training Data (On-policy from trained policy)
    # --------------------------------------------------------

    print("Filling replay buffer...")

    states = torch.zeros((2048, state_dim), device=device)

    with torch.no_grad():
        for step in range(10000):
            # Extract state components
            x = states[:, 0]
            y = states[:, 1]
            vx = states[:, 2]
            vy = states[:, 3]
            theta = states[:, 4]
            omega = states[:, 5]
            
            # Get reference trajectory
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(step * dt)
            
            # Build observation for policy
            obs = torch.stack([
                pos_ref[:, 0] - x,
                pos_ref[:, 1] - y,
                vel_ref[:, 0] - vx,
                vel_ref[:, 1] - vy,
                acc_ref[:, 0],
                acc_ref[:, 1],
                torch.sin(theta),
                torch.cos(theta),
                omega
            ], dim=1)
            
            # Get actions from policy
            actions = policy(obs)
            # actions = torch.tensor([[5.0, 5.0]], device=device)
            next_states = drone.step(states, actions, control_mode=cm)

            buffer.add(states, actions, next_states)
            states = next_states

    print("Replay buffer size:", buffer.size)

    # --------------------------------------------------------
    # 5. Train World Model (ONE-STEP ONLY)
    # --------------------------------------------------------

    epochs = 100
    batch_size = 4096
    steps_per_epoch = 500
    steps_to_predict = 2

    print("Starting WM training...")

    for epoch in range(epochs):

        epoch_loss = 0.0

        for _ in range(steps_per_epoch):

            # Sample initial batch
            s, a, s_next = buffer.sample(batch_size)

            total_loss = 0.0

            # First step (teacher forcing)
            s_norm = state_norm.encode(s)
            a_norm = action_norm.encode(a)
            s_next_norm = state_norm.encode(s_next)

            wm_input = torch.cat([s_norm, a_norm], dim=-1)
            pred_next_norm = world_model(wm_input)

            loss = F.mse_loss(pred_next_norm, s_next_norm)
            total_loss += loss

            # Now autoregressive steps
            current_state = pred_next_norm.detach()   # detach to stabilize early training
            # current_state = pred_next_norm   # detach to stabilize early training

            for k in range(1, steps_to_predict):

                # Sample a new random action
                # (since replay buffer isn't sequential)
                _, next_a, next_s_next = buffer.sample(batch_size)

                a_norm = action_norm.encode(next_a)
                target_norm = state_norm.encode(next_s_next)

                wm_input = torch.cat([current_state, a_norm], dim=-1)

                pred_next_norm = world_model(wm_input)

                step_loss = F.mse_loss(pred_next_norm, target_norm)

                total_loss += step_loss

                current_state = pred_next_norm.detach()
                # current_state = pred_next_norm

            total_loss = total_loss / steps_to_predict

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | WM Loss: {epoch_loss:.6f}")

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    print(f"Saving in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(world_model.state_dict(), os.path.join(output_dir, "world_model.pt"))
    torch.save(
        {'mean': state_norm.mean, 'std': state_norm.std},
        os.path.join(output_dir, "state_norm.pt")
    )
    torch.save(
        {'mean': action_norm.mean, 'std': action_norm.std},
        os.path.join(output_dir, "action_norm.pt")
    )

    print("World model saved.")


# ============================================================
# Evaluate
# ============================================================


def evaluate_rollout(world_model, drone, traj_gen, state_norm, action_norm, device, cm, policy):

    world_model.eval()
    policy.eval()
    dt = 0.01
    rollout_steps = 300
    num_envs = 1

    with torch.no_grad():

        # Start from real state
        state_real = torch.zeros((num_envs, 6), device=device)
        state_model = state_real.clone()

        real_traj = []
        model_traj = []
        action_traj = []
        target_traj = []

        for t in range(rollout_steps):

            pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)
            pos_ref, vel_ref, acc_ref = pos_ref.squeeze(0), vel_ref.squeeze(0), acc_ref.squeeze(0)

            # Build observation from current state
            x, y, vx, vy, theta, omega = state_real.squeeze(0)
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
            
            # Get action from policy
            action = policy(obs)
            # action = torch.tensor([[5.0, 5.0]], device=device)

            # Real next state
            next_real = drone.step(state_real, action, control_mode=cm)

            # Model prediction
            s_norm = state_norm.encode(state_model)
            a_norm = action_norm.encode(action)
            wm_input = torch.cat([s_norm, a_norm], dim=-1)

            pred_next_norm = world_model(wm_input)
            pred_next = pred_next_norm * state_norm.std + state_norm.mean

            # Store
            real_traj.append(next_real.cpu())
            model_traj.append(pred_next.cpu())
            action_traj.append(action.cpu())
            target_traj.append(pos_ref.cpu())

            # Step forward
            state_real = next_real
            state_model = pred_next

        real_traj = torch.cat(real_traj)
        model_traj = torch.cat(model_traj)
        action_traj = torch.cat(action_traj)

        return real_traj, model_traj, action_traj, target_traj

# ============================================================
# Main
# ============================================================

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def main(cfg: DictConfig):

    cm = "srt"
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm + "_copy")
    print(output_dir)
    start = time.time()
    train(cm, cfg)
    print("Total time:", time.time() - start)


    device = "cpu"
    state_norm = Normalizer(6, device)
    action_norm = Normalizer(ACT_DIMS[cm], device)
    drone = BicopterDynamics(cfg=cfg, device=device)
    target_gen = RandomTrajectoryGenerator(num_envs=1, device=device)
    
    # Load policy
    policy = MLP(input=9, hidden=64, output=ACT_DIMS[cm]).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    policy.eval()
    
    state_norm.load(os.path.join(output_dir, "state_norm.pt"))
    action_norm.load(os.path.join(output_dir, "action_norm.pt"))

    world_model = MLP(input=6 + ACT_DIMS[cm], hidden=256, output=6).to(device)
    world_model.load_state_dict(torch.load(os.path.join(output_dir, "world_model.pt"), map_location=device))

    real_traj, model_traj, action_traj, target_traj = evaluate_rollout(
    world_model, drone, target_gen, state_norm, action_norm, device, cm, policy)

    plt.figure(figsize=(12, 8))
    state_labels = ["x", "y", "vx", "vy", "theta", "omega"]
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(real_traj[:, i], label="real", linewidth=2)
        plt.plot(model_traj[:, i], label="model", linewidth=2)
        plt.legend()
        plt.title(f"Rollout comparison: {state_labels[i]}")
        plt.xlabel("timestep")
    
    plt.tight_layout()
    plt.show()

    print(torch.nn.functional.mse_loss(real_traj, model_traj))

    

    renderer = MultiTrajectoryRenderer(drone=drone, video_path=None)
    renderer.add_agent(real_traj, target_traj, action_traj, cm, (0, 0, 255), "REAL")
    renderer.add_agent(model_traj, target_traj, action_traj, cm, (255, 0, 0), "WORLD_MODEL")

    renderer.run()


if __name__ == "__main__":
    ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}
    main()
