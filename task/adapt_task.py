from env.adapt_env import TrackingEnv
from algo.AM import AM
from utils.nn import *
import torch
import os

ACT_DIMS = {"srt": 2, "ctbr": 2, "lv": 5}

def train(cfg):
    device = torch.device(cfg.task.device if torch.cuda.is_available() else "cpu")
    epochs = cfg.task.epochs
    
    # 1. Setup Environment
    env = TrackingEnv(cfg, cfg.task.num_envs, device)
    
    # 2. Setup Policy & Optimizer
    policy = BicopterPolicy(obs_dim=11, act_dim=ACT_DIMS[cfg.cm]).to(device) # example act_dim
    encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)
    adaptor = AdaptationModule(input_dim=(6+ACT_DIMS[cfg.cm]), z_dim=2, k=cfg.task.k).to(device)
    
    # 3. Setup Algorithm
    trainer = AM(cfg, policy, encoder, adaptor)

    # --- PHASE 1 ---
    opt1 = torch.optim.Adam(list(policy.parameters()) + list(encoder.parameters()), lr=1e-3)
    for epoch in range(epochs):
        loss = trainer.train_control(env, opt1)
        if epoch % 10 == 0: print(f"Phase 1 Epoch {epoch} | Loss: {loss:.4f}")

    # --- PHASE 2 ---
    # Freeze policy and encoder
    policy.eval(); encoder.eval()
    opt2 = torch.optim.Adam(adaptor.parameters(), lr=1e-3)
    for epoch in range(epochs):
        loss = trainer.train_adaptation(env, opt2)
        if epoch % 10 == 0: print(f"Phase 2 Epoch {epoch} | Loss: {loss:.4f}")

    # Save everything
    save_dir = os.path.join("outputs", cfg.cm)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(save_dir, "policy.pt"))
    torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
    torch.save(adaptor.state_dict(), os.path.join(save_dir, "adaptor.pt"))