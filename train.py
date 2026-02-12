import os
import torch
import matplotlib.pyplot as plt

from dynamics.bicopter_dynamics import BicopterDynamics

from utils.nn import BicopterPolicy
from utils.randomizer import scale_randomization
from utils.rand_traj_gen import RandomTrajectoryGenerator


import hydra
from omegaconf import DictConfig

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def train(cfg: DictConfig):
    """
    ----------------------------------------------------------------------
    Main training function
    ----------------------------------------------------------------------
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    num_envs = 1 if device == "cpu" else  2048
    print(f"num_envs: {num_envs}")
    epochs = 200
    steps  = 500
    horizon = 50 
    dt = 0.01

    ACT_DIMS = {
        "srt": 2,  # T1, T2
        "ctbr": 2, # T, omega
        "lv": 5    #  vx, vy, kv, kR, kw  
    }

    #================
    # Control mode:
    cm = "lv" 
    #================

    drone = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
    policy = BicopterPolicy(
        obs_dim=9, 
        act_dim=ACT_DIMS[cm]
    ).to(device)


    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    loss_history = []

    for epoch in range(epochs):
        traj_gen.reset()
        print(scale_randomization(cfg))

        states = torch.zeros((num_envs, 8), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0
        
        epoch_loss = 0.0
        num_chunks = 0
        
        for chunk_start in range(0, steps, horizon):
            chunk_end = min(chunk_start + horizon, steps)
            
            optimizer.zero_grad()
            
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
                    acc_ref[:, 0].expand_as(e_px),
                    acc_ref[:, 1].expand_as(e_py),
                    torch.sin(theta),
                    torch.cos(theta),
                    omega
                ], dim=1)

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

        if epoch % 100 == 0 or epoch == (epochs-1):
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")


    # Plot the loss
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    # Export the model
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, f"{cm}3.pt"))
    print("Policy saved!")


if __name__ == "__main__":
    train()
