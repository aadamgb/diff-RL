import os
import torch
from omegaconf import DictConfig
from dynamics.bicopter_dynamics import BicopterDynamics
from utils.nn import BicopterPolicy
from utils.randomizer import env_randomization
from utils.rand_traj_gen import RandomTrajectoryGenerator

ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def train(cfg: DictConfig):
    cm = cfg.cm
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_envs = cfg.num_envs if device.type == "cuda" else 1
    
    save_dir = os.path.join("outputs", cm)

    drone = BicopterDynamics(device=device, cfg=cfg)
    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cm]).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    for epoch in range(cfg.epochs):
        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)
        
        states = torch.zeros((num_envs, 6), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0
        
        # Initial Targets
        pos_ref, vel_ref, acc_ref = traj_gen.get_hover_targets(boundary=5.0)
        
        epoch_loss = 0.0
        timer = torch.zeros(num_envs, device=device)
        
        # Lists to store horizon data
        buffer_states = []
        buffer_refs = [] # FIX: Store references per-step
        
        for t in range(cfg.steps):
            # 1. Check for Reset (Logic looks good!)
            dist = torch.sqrt(((pos_ref - states[:, :2])**2).sum(dim=1))
            close_envs = dist < 0.5 
            timer[close_envs] += cfg.dt
            timer[~close_envs] = 0.0

            reset_pos = timer > 3.0 # Hover for 3 seconds
            if reset_pos.any():
                new_p, new_v, new_a = traj_gen.get_hover_targets(boundary=5.0)
                pos_ref[reset_pos] = new_p[reset_pos]
                vel_ref[reset_pos] = new_v[reset_pos]
                acc_ref[reset_pos] = new_a[reset_pos]
                timer[reset_pos] = 0.0

            # 2. Build Observation
            x, y, vx, vy, theta, omega = states.unbind(dim=1)
            obs = torch.stack([
                pos_ref[:, 0]-x, pos_ref[:, 1]-y, 
                vel_ref[:, 0]-vx, vel_ref[:, 1]-vy, 
                acc_ref[:, 0], acc_ref[:, 1], 
                torch.sin(theta), torch.cos(theta), omega
            ], dim=1)

            # 3. Action and Step
            actions = policy(obs)
            states = drone.step(states, actions, control_mode=cm)
            
            # 4. Store for Backprop
            buffer_states.append(states)
            buffer_refs.append(pos_ref.clone()) # Store the reference AT THIS STEP

            # 5. Horizon Optimization
            if (t + 1) % cfg.horizon == 0:
                traj_chunk = torch.stack(buffer_states) # (H, N, 6)
                ref_chunk = torch.stack(buffer_refs)    # (H, N, 2)
                
                pos_error = torch.mean((traj_chunk[..., :2] - ref_chunk)**2)
                vel_penalty = torch.mean(traj_chunk[..., 2:4]**2)
                rate_penalty = torch.mean(traj_chunk[..., 5]**2)
                
                loss = 1.0 * pos_error + 0.25 * vel_penalty + 0.1 * rate_penalty
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                states = states.detach()
                epoch_loss += loss.item()
                buffer_states, buffer_refs = [], []

        if epoch % 10 == 0: print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")

    torch.save(policy.state_dict(), os.path.join(save_dir, "pc_policy.pt"))