from env.race_env import RaceEnv
from algo.TBPTTv2 import TBPTT
from utils.nn import BicopterPolicy
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def plot_track(gates):
    """Plot gate positions with look_at vectors"""
    fig, ax = plt.subplots(figsize=(15, 10))
    
    positions = np.array([gate['position'] for gate in gates])
    look_ats = np.array([gate['look_at'] for gate in gates])
    
    # Plot gate positions
    ax.scatter(positions[:, 0], positions[:, 1], c='red', s=100, label='Gates', zorder=5)
    
    # Plot look_at vectors
    for i, (pos, look_at) in enumerate(zip(positions, look_ats)):
        # Normalize look_at vector to unit vector for visualization
        look_at_norm = np.array(look_at) / np.linalg.norm(look_at)
        # Scale for visibility
        scale = 1.5
        ax.arrow(pos[0], pos[1], look_at_norm[0] * scale, look_at_norm[1] * scale,
                head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.7)
    
    # Connect gates with a line
    ax.plot(positions[:, 0], positions[:, 1], 'r--', alpha=0.3)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('A2RL Track with Gate Positions and Look-at Vectors')
    ax.set_xlim(2, 25)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def train(cfg):
    # plot_track(cfg.track.gates)
    
    device = torch.device(cfg.task.device if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    env = RaceEnv(cfg, cfg.task.num_envs, device)
    
    # 2. Setup Policy & Optimizer
    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cfg.cm]).to(device) # example act_dim
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # # 3. Setup Algorithm
    trainer = TBPTT(policy, optimizer, cfg)

    # # 4. Training Loop
    for epoch in range(cfg.task.epochs):
        loss = trainer.train_epoch(env)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    # 5. Save
    save_path = os.path.join("outputs", cfg.cm, "race_policy.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
