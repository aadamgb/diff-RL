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

def train_P1(cm, cfg: DictConfig):
    """
    ----------------------------------------------------------------------
    PHASE 1
    ----------------------------------------------------------------------
    """
    print("=== Starting Phase 1 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    num_envs = 1 if device == "cpu" else   1024 * 2
    print(f"num_envs: {num_envs}")
    epochs = 100
    steps  = 300
    horizon = 50 
    dt = 0.01

    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
    policy = BicopterPolicy(obs_dim=11, act_dim=ACT_DIMS[cm]).to(device)
    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)

    optimizer = torch.optim.Adam(list(policy.parameters()) +
                                 list(env_encoder.parameters()), lr=1e-3)

    loss_history = []
    for epoch in range(epochs):
        # Generate random target trajectory #
        traj_gen.reset()

        # Randomize the environmental parameters per num_envs #
        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)
        e = torch.stack([env_params["m"], env_params["J"], env_params["l"], env_params["C_Dx"], env_params["C_Dy"]], dim=1)
        
        # Initialize the bicopter state #
        states = torch.zeros((num_envs, 6), device=device)
        # randomize initial positon
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0
                      
        epoch_loss = 0.0
        num_chunks = 0
        
        for chunk_start in range(0, steps, horizon):
            chunk_end = min(chunk_start + horizon, steps)
            
            optimizer.zero_grad()
            z = env_encoder(e)
            
            # Simulate one chunk
            chunk_traj, chunk_target_pos, chunk_target_vel = [], [], []
            
            for t in range(chunk_start, chunk_end):
                x, y, vx, vy, theta, omega = states.unbind(dim=1)
                pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)

                # Compute errors
                e_px = pos_ref[:, 0] - x
                e_py = pos_ref[:, 1] - y
                e_vx = vel_ref[:, 0] - vx
                e_vy = vel_ref[:, 1] - vy

                # Build observation input
                obs = torch.stack([
                    e_px, e_py, e_vx, e_vy,
                    acc_ref[:, 0],
                    acc_ref[:, 1],
                    torch.sin(theta),
                    torch.cos(theta),
                    omega
                ], dim=1)

                obs = torch.cat([obs, z], dim=1)  # added the ecoded env_params

                # Update state
                actions = policy(obs)
                states = drone.step(states, actions, control_mode=cm)

                chunk_traj.append(states)
                chunk_target_pos.append(pos_ref)
                chunk_target_vel.append(vel_ref)

            # Stack into (horizon, num_envs, 6)
            traj_chunk = torch.stack(chunk_traj)
            target_pos_chunk = torch.stack(chunk_target_pos).unsqueeze(1)
            target_vel_chunk = torch.stack(chunk_target_vel).unsqueeze(1)
            
            # Compute loss for this chunk
            pos_error = torch.mean(torch.sum((traj_chunk[..., :2] - target_pos_chunk)**2, dim=1))
            vel_error = torch.mean(torch.sum((traj_chunk[..., 2:4] - target_vel_chunk)**2, dim=1))
            rate_penalty = torch.mean(traj_chunk[..., 5]**2)
            
            loss = 1.0 * pos_error + 1.0 * vel_error + 0.25 * rate_penalty
            epoch_loss += loss.item()
            
            # Backprop and step for this chunk
            loss.backward()
            optimizer.step()
            
            states = states.detach()  # This prevents gradients from flowing through previous chunks
            num_chunks += 1
        
        # Average loss over chunks for logging
        avg_loss = epoch_loss / num_chunks
        loss_history.append(avg_loss / num_envs)

        if epoch % 10 == 0:
            print(f"(Phase 1) Epoch {epoch} | Loss = {avg_loss:.3f}")


    # Plot the loss
    # plt.figure()
    # plt.plot(loss_history)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss")
    # plt.grid(True)
    # plt.show()

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    torch.save(env_encoder.state_dict(), os.path.join(output_dir, "encoder.pt"))
    print("Phase 1 Completed | Policy and Encoder saved!\n")



def train_P2(dt,
            steps,
            k,
            epochs,
            device,
            num_envs,
            cm="ctbr",
            cfg=DictConfig):
    """
    ----------------------------------------------------------------------
    PHASE 2
    ----------------------------------------------------------------------
    """
    print("=== Starting Phase 2 ===")
    print(f"num_envs: {num_envs}")
     # ---- Load PHASE 1 models (actor and encoder) ----
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)

    policy = BicopterPolicy(obs_dim=11, act_dim=ACT_DIMS[cm]).to(device)
    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)

    policy.load_state_dict(torch.load(os.path.join(output_dir, "policy.pt"), map_location=device))
    env_encoder.load_state_dict(torch.load(os.path.join(output_dir, "encoder.pt"), map_location=device))

    print(os.path.join(output_dir, "policy.pt"))

    # Gradients should not flow here, these are loaded to generate a refernce z for the adapt module
    policy.eval()
    env_encoder.eval()

    for p in policy.parameters():
        p.requires_grad = False
    for p in env_encoder.parameters():
        p.requires_grad = False

    # ---- Adaptation module ----
    adapt_module = AdaptationModule(input_dim=(6 + ACT_DIMS[cm]), z_dim=2, k=k).to(device)
    optimizer = torch.optim.Adam(adapt_module.parameters(), lr=1e-3)

    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    for epoch in range(epochs):

        traj_gen.reset()

        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)

        e = torch.stack([env_params["m"], env_params["J"], env_params["l"], env_params["C_Dx"], env_params["C_Dy"]], dim=1)

        z_true = env_encoder(e).detach()   # detach because encoder is frozen
        states = torch.zeros((num_envs, 6), device=device)
        history = torch.zeros((num_envs, k, (6 + ACT_DIMS[cm])), device=device)

        total_loss = 0.0
        for t in range(steps):

            x, y, vx, vy, theta, omega = states.unbind(dim=1)
            pos_ref, vel_ref, acc_ref = traj_gen.get_target(t * dt)

            e_px = pos_ref[:, 0] - x
            e_py = pos_ref[:, 1] - y
            e_vx = vel_ref[:, 0] - vx
            e_vy = vel_ref[:, 1] - vy

            obs_base = torch.stack([
                e_px, e_py, e_vx, e_vy,
                acc_ref[:, 0],
                acc_ref[:, 1],
                torch.sin(theta),
                torch.cos(theta),
                omega
            ], dim=1)

            # POLICY USES TRUE Z
            obs = torch.cat([obs_base, z_true], dim=1)

            actions = policy(obs)
            states = drone.step(states, actions, control_mode=cm)

            # Update history
            sa = torch.cat([states, actions], dim=1)  # (B, 8)
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = sa

            z_hat = adapt_module(history)

            loss = torch.mean((z_hat - z_true) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"(Phase 2) Epoch {epoch} | Loss {total_loss:.3f}")

    torch.save(adapt_module.state_dict(), os.path.join(output_dir, "adapt_module.pt"))
    print("Phase 2 Completed | Adaptation module saved!")
            


@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def main(cfg: DictConfig):
    cm = "srt" 
    dt = 0.01
    steps = 300
    horizon = 20
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 1024 if device.type == "cuda" else 1

    # PHASE 1: Train policy pi, and econder mu
    start_time = time.time()
    train_P1(cm=cm, cfg=cfg)
    end_time = time.time()
    training_duration_1 = end_time - start_time
    
    # PHASE 2: Train adaptation module
    start_time = time.time()
    train_P2(dt=dt,
                steps=steps,
                k=horizon,
                epochs=epochs,
                device=device,
                num_envs=num_envs,
                cm=cm,
                cfg=cfg)
    end_time = time.time()
    training_duration_2 = end_time - start_time

    # Prints and training results
    total_duration = training_duration_1 + training_duration_2
    print("\n=== Training Times ===")
    print(f"Phase 1 training time: {int(training_duration_1 // 3600)}h {int((training_duration_1 % 3600) // 60)}m {training_duration_1 % 60:.2f}s ")
    print(f"Phase 2 training time: {int(training_duration_2 // 3600)}h {int((training_duration_2 % 3600) // 60)}m {training_duration_2 % 60:.2f}s ")
    print(f"Total training time: {int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {total_duration % 60:.2f}s ")
    

if __name__ == "__main__":

    ACT_DIMS = {
        "srt": 2,
        "ctbr": 2,
        "lv": 5
    }
    
    main()