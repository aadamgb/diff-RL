import os
import torch
import matplotlib.pyplot as plt
import time

from dynamics.bicopter_dynamics import BicopterDynamics
from utils.nn import *
from utils.randomizer import env_randomization
from utils.rand_traj_gen import RandomTrajectoryGenerator

import hydra
from omegaconf import DictConfig

# --- Helper Class for Normalization ---
class Normalizer:
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.std = torch.ones(dim, device=device)
        self.device = device

    def update(self, data):
        # Data shape expected: [Total_Samples, Dim]
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-6

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return (x * self.std) + self.mean

def train(cm, cfg: DictConfig):
    print("=== Starting Training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 1 if device.type == "cpu" else 2048
    print(f"num_envs = {num_envs}")
    epochs  = 200
    steps   = 600
    horizon = 100
    dt      = 0.01
    state_dim = 6

    # -------------------------------------------------
    # Environment & Generators
    # -------------------------------------------------
    drone    = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    # -------------------------------------------------
    # Normalization Calibration
    # -------------------------------------------------
    # We run a quick random rollout to get data statistics
    print("Calibrating Normalizers...")
    state_norm = Normalizer(state_dim, device)
    action_norm = Normalizer(ACT_DIMS[cm], device)

    with torch.no_grad():
        temp_states = torch.randn((num_envs * 10, state_dim), device=device)
        temp_actions = torch.randn((num_envs * 10, ACT_DIMS[cm]), device=device)
        state_norm.update(temp_states)
        action_norm.update(temp_actions)

    # -------------------------------------------------
    # Models
    # -------------------------------------------------
    policy = MLP(input=9, hidden=64, output=ACT_DIMS[cm]).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # World model inputs: normalized state + normalized action
    world_model = MLP(input=state_dim + ACT_DIMS[cm], hidden=128, output=state_dim).to(device)
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

    # =================================================
    # TRAINING LOOP
    # =================================================
    for epoch in range(epochs):
        traj_gen.reset()
        states = torch.zeros((num_envs, state_dim), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0 

        epoch_policy_loss = 0.0
        epoch_wm_loss     = 0.0

        for horizon_start in range(0, steps, horizon):
            horizon_states  = []
            horizon_actions = []
            horizon_pos_ref = []
            horizon_vel_ref = []

            # ------------------------------------------
            # Real Environment Rollout (Policy)
            # ------------------------------------------
            for t in range(horizon_start, horizon_start + horizon):
                x, y, vx, vy, theta, omega = states.unbind(dim=1)
                pos_ref, vel_ref, acc_ref  = traj_gen.get_target(t * dt)

                obs = torch.stack([
                    pos_ref[:,0] - x, pos_ref[:,1] - y,
                    vel_ref[:,0] - vx, vel_ref[:,1] - vy,
                    acc_ref[:,0], acc_ref[:,1],
                    torch.sin(theta), torch.cos(theta), omega
                ], dim=1)

                actions = policy(obs)
                states  = drone.step(states, actions, control_mode=cm)

                horizon_states.append(states)
                horizon_actions.append(actions)
                horizon_pos_ref.append(pos_ref)
                horizon_vel_ref.append(vel_ref)

            # ------------------------------------------
            # Policy Loss
            # ------------------------------------------
            traj_horizon = torch.stack(horizon_states) 
            target_pos = torch.stack(horizon_pos_ref)
            target_vel = torch.stack(horizon_vel_ref)

            pos_error = ((traj_horizon[..., :2] - target_pos)**2).mean()
            vel_error = ((traj_horizon[..., 2:4] - target_vel)**2).mean()
            policy_loss = pos_error + vel_error

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            states = states.detach()
            epoch_policy_loss += policy_loss.item()

            # ------------------------------------------
            # World Model Training (Multi-step + Normalization)
            # ------------------------------------------
            wm_steps = 50
            if horizon > wm_steps:
                # Random start point for sequence training
                start_idx = torch.randint(0, horizon - wm_steps, (1,)).item()
                
                # Detach initial state to isolate World Model gradients from Policy
                current_state = horizon_states[start_idx].detach()
                seq_wm_loss = 0.0
                
                for step in range(wm_steps):
                    # 1. Normalize inputs
                    norm_s = state_norm.encode(current_state)
                    norm_a = action_norm.encode(horizon_actions[start_idx + step].detach())
                    
                    wm_in = torch.cat([norm_s, norm_a], dim=-1)
                    
                    # 2. Predict next state (normalized)
                    norm_pred_next_state = world_model(wm_in)
                    
                    # 3. Decode to real-world units
                    pred_next_state = state_norm.decode(norm_pred_next_state)
                    
                    # 4. Compute Loss against ground truth
                    target_state = horizon_states[start_idx + step + 1].detach()
                    seq_wm_loss += torch.nn.functional.mse_loss(pred_next_state, target_state)
                    
                    # 5. Auto-regressive step: use prediction as next input
                    current_state = pred_next_state

                # Average loss over the sequence length
                seq_wm_loss = seq_wm_loss / wm_steps

                wm_optimizer.zero_grad()
                seq_wm_loss.backward()
                wm_optimizer.step()
                
                epoch_wm_loss += seq_wm_loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Policy Loss: {epoch_policy_loss:.4f} | WM Loss: {epoch_wm_loss:.6f}")

    # Save models
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    torch.save(world_model.state_dict(), os.path.join(output_dir, "world_model.pt"))
    print(output_dir)
    print("Models saved!\n")

    state_norm_data = {'mean': state_norm.mean, 'std': state_norm.std} 
    action_norm_data = {'mean': action_norm.mean, 'std': action_norm.std}      
    torch.save(state_norm_data, os.path.join(output_dir, "state_norm.pt"))
    torch.save(action_norm_data, os.path.join(output_dir, "action_norm.pt"))

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def main(cfg: DictConfig):
    cm = "srt" 
    start_time = time.time()
    train(cm=cm, cfg=cfg)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Total time: {int(training_duration // 3600)}h {int((training_duration % 3600) // 60)}m {training_duration % 60:.2f}s ")

if __name__ == "__main__":
    ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}
    main()