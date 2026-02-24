import os
import torch
from omegaconf import DictConfig

from dynamics.bicopter_dynamics import BicopterDynamics
from utils.nn import *
from utils.rand_traj_gen import RandomTrajectoryGenerator

ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def train(cfg: DictConfig):
    cm = cfg.cm
    device =  torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = cfg.num_envs if device.type == "cuda" else 1

    dir = os.path.join("outputs", cm)

    epochs, steps, dt = cfg.epochs, cfg.steps, cfg.dt

    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cm]).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(epochs):
        states = torch.zeros((num_envs, 6), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0

        pos_ref, vel_ref, acc_ref = traj_gen.get_hover_targets(boundary=5.0)
        
        epoch_loss = 0.0
        timer = torch.zeros(num_envs, device=device)
        trajectory = []
        
        for t in range(steps):
            distance = torch.sqrt(((pos_ref - states[:, :2])**2).sum(dim=1))
            
            close_envs = distance < 0.5
            timer[close_envs] += dt
            timer[~close_envs] = 0.0

            reset_pos = timer > 1.0
            if reset_pos.any():
                # Generate new targets for all envs
                new_pos_ref, new_vel_ref, new_acc_ref = traj_gen.get_hover_targets(boundary=5.0)
                
                # Get indices of environments that need resetting
                reset_indices = torch.where(reset_pos)[0]
                
                # Assign new targets to resetting environments (cycle through generated targets)
                pos_ref[reset_pos] = new_pos_ref[reset_indices % num_envs]
                vel_ref[reset_pos] = new_vel_ref[reset_indices % num_envs]
                acc_ref[reset_pos] = new_acc_ref[reset_indices % num_envs]
                timer[reset_pos] = 0.0

            x, y, vx, vy, theta, omega = states.unbind(dim=1)
            obs = torch.stack([
                pos_ref[:, 0]-x, pos_ref[:, 1]-y, 
                vel_ref[:, 0]-vx, vel_ref[:, 1]-vy, 
                acc_ref[:, 0], acc_ref[:, 1], 
                torch.sin(theta), torch.cos(theta), omega
            ], dim=1)

            actions = policy(obs)
            states = drone.step(states, actions, control_mode=cm)
            trajectory.append(states.clone())

            # Optimization in horizions
            if (t + 1) % cfg.horizon == 0:
                # Compute loss
                traj_horizon = torch.stack(trajectory)  # (horizon, num_envs, 6)
                
                pos_error = torch.mean((traj_horizon[..., :2] - pos_ref.unsqueeze(0))**2)
                vel_penalty = torch.mean(traj_horizon[..., 2:4]**2)
                rate_penalty = torch.mean(traj_horizon[..., 5]**2)
                
                # Combined loss
                loss = 1.0 * pos_error + 0.25 * vel_penalty + 0.25 * rate_penalty
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                states = states.detach()
                epoch_loss += loss.item()
                trajectory = []  # Clear trajectory for next horizon


        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} (pos: {pos_error:.4f}, vel: {vel_penalty:.4f}, rate: {rate_penalty:.4f})")
    
    # Save trained policy
    os.makedirs(dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dir, "pc_policy.pt"))
    print(f"\nTraining completed! Policy saved to {dir}/pc_policy.pt")


