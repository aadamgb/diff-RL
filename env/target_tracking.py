import os
import torch
import time
from omegaconf import DictConfig

from dynamics.bicopter_dynamics import BicopterDynamics
from utils.nn import BicopterPolicy, IntrinsicsEncoder, AdaptationModule
from utils.randomizer import env_randomization
from utils.rand_traj_gen import RandomTrajectoryGenerator

# Mapping for action dimensions
ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def train(cfg: DictConfig):
    cm = cfg.cm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = cfg.num_envs if device.type == "cuda" else 1
    
    # PHASE 1
    start_p1 = time.time()
    train_P1(cm, cfg, device, num_envs)
    dur_p1 = time.time() - start_p1

    # PHASE 2
    start_p2 = time.time()
    train_P2(cfg.dt, cfg.steps, cfg.env.k, cfg.epochs, device, num_envs, cm, cfg)
    dur_p2 = time.time() - start_p2

    print(f"\nTotal Training Time: {int((dur_p1 + dur_p2) // 60)}m {(dur_p1 + dur_p2) % 60:.2f}s")

def train_P1(cm, cfg, device, num_envs):
    print("=== Starting Phase 1 ===")
    epochs, steps, horizon, dt = cfg.epochs, cfg.steps, cfg.horizon, cfg.dt
    
    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
    policy = BicopterPolicy(obs_dim=11, act_dim=ACT_DIMS[cm]).to(device)
    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)

    optimizer = torch.optim.Adam(list(policy.parameters()) + list(env_encoder.parameters()), lr=1e-3)

    for epoch in range(epochs):
        traj_gen.reset()
        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)
        e = torch.stack([env_params["m"], env_params["J"], env_params["l"], env_params["C_Dx"], env_params["C_Dy"]], dim=1)
        
        states = torch.zeros((num_envs, 6), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0
                      
        epoch_loss = 0.0
        for chunk_start in range(0, steps, horizon):
            chunk_end = min(chunk_start + horizon, steps)
            optimizer.zero_grad()
            z = env_encoder(e)
            
            chunk_traj, chunk_target_pos, chunk_target_vel = [], [], []
            for t in range(chunk_start, chunk_end):
                x, y, vx, vy, theta, omega = states.unbind(dim=1)
                pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)

                obs = torch.cat([torch.stack([pos_ref[:, 0]-x, pos_ref[:, 1]-y, vel_ref[:, 0]-vx, vel_ref[:, 1]-vy, 
                                             acc_ref[:, 0], acc_ref[:, 1], torch.sin(theta), torch.cos(theta), omega], dim=1), z], dim=1)

                actions = policy(obs)
                states = drone.step(states, actions, control_mode=cm)
                chunk_traj.append(states)
                chunk_target_pos.append(pos_ref)
                chunk_target_vel.append(vel_ref)

            traj_chunk = torch.stack(chunk_traj)
            target_pos_chunk = torch.stack(chunk_target_pos).unsqueeze(1)
            target_vel_chunk = torch.stack(chunk_target_vel).unsqueeze(1)

            pos_error = torch.mean(torch.sum((traj_chunk[..., :2] - target_pos_chunk)**2, dim=1))
            vel_error = torch.mean(torch.sum((traj_chunk[..., 2:4] - target_vel_chunk)**2, dim=1))
            rate_penalty = torch.mean(traj_chunk[..., 5]**2)

            loss = 1.0 * pos_error + 1.0 * vel_error + 0.25 * rate_penalty
            epoch_loss += loss.item()
            
            # Backprop and step for this chunk
            loss.backward()
            optimizer.step()
            
            states = states.detach()  # This prevents gradients from flowing through previous chunks
        
        if epoch % 10 == 0: print(f"Phase 1 | Epoch {epoch} | Loss: {epoch_loss/cfg.num_envs:.3f}")

    output_dir = os.path.join("outputs", cm)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    torch.save(env_encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))

def train_P2(dt, steps, k, epochs, device, num_envs, cm, cfg):
    print("\n=== Starting Phase 2 ===")
    output_dir = os.path.join("outputs", cm)
    
    policy = BicopterPolicy(obs_dim=11, act_dim=ACT_DIMS[cm]).to(device)
    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)
    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    env_encoder.load_state_dict(torch.load(os.path.join(output_dir, "encoder.pt"), map_location=device))

    policy.eval(); env_encoder.eval()
    for p in policy.parameters(): p.requires_grad = False
    for p in env_encoder.parameters(): p.requires_grad = False

    adapt_module = AdaptationModule(input_dim=(6 + ACT_DIMS[cm]), z_dim=2, k=k).to(device)
    optimizer = torch.optim.Adam(adapt_module.parameters(), lr=1e-3)
    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    for epoch in range(epochs):
        traj_gen.reset()
        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)
        e = torch.stack([env_params["m"], env_params["J"], env_params["l"], env_params["C_Dx"], env_params["C_Dy"]], dim=1)
        z_true = env_encoder(e).detach()
        
        states = torch.zeros((num_envs, 6), device=device)
        history = torch.zeros((num_envs, k, (6 + ACT_DIMS[cm])), device=device)

        total_loss = 0.0
        for t in range(steps):
            x, y, vx, vy, theta, omega = states.unbind(dim=1)
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)
            obs = torch.cat([torch.stack([pos_ref[:, 0]-x, pos_ref[:, 1]-y, vel_ref[:, 0]-vx, vel_ref[:, 1]-vy, 
                                         acc_ref[:, 0], acc_ref[:, 1], torch.sin(theta), torch.cos(theta), omega], dim=1), z_true], dim=1)

            actions = policy(obs)
            states = drone.step(states, actions, control_mode=cm)
            
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = torch.cat([states, actions], dim=1)
            
            z_hat = adapt_module(history)
            loss = torch.mean((z_hat - z_true) ** 2)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0: print(f"Phase 2 | Epoch {epoch} | Loss {total_loss:.3f}")
    
    torch.save(adapt_module.state_dict(), os.path.join(output_dir, "adapt_module.pt"))