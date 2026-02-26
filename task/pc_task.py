from env.pc_env import BicopterHoverEnv
from algo.TBPTT import TBPTT
from utils.nn import BicopterPolicy
import torch
import os

ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def train(cfg):
    device = torch.device(cfg.task.device if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Environment
    env = BicopterHoverEnv(cfg, cfg.task.num_envs, device)
    
    # 2. Setup Policy & Optimizer
    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cfg.cm]).to(device) # example act_dim
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # 3. Setup Algorithm
    trainer = TBPTT(policy, optimizer, cfg)

    # 4. Training Loop
    for epoch in range(cfg.task.epochs):
        loss = trainer.train_epoch(env)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    # 5. Save
    save_path = os.path.join("outputs", cfg.cm, "pc_policy.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)