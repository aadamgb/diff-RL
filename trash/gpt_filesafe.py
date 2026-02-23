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

def train(cm, cfg: DictConfig):

    print("=== Starting Phase 1 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 1 if device.type == "cpu" else 2048

    epochs  = 200
    steps   = 300
    horizon = 50
    dt      = 0.01

    # -------------------------------------------------
    # Environment
    # -------------------------------------------------
    drone    = BicopterDynamics(device=device, cfg=cfg)
    traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)

    # -------------------------------------------------
    # Policy
    # -------------------------------------------------
    policy = BicopterPolicy(obs_dim=9, act_dim=ACT_DIMS[cm]).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # -------------------------------------------------
    # Deterministic World Model
    # -------------------------------------------------
    obs_encoder   = MLP(input=6, hidden=64, output=32).to(device)
    sequence_model = nn.GRU(
        input_size=32 + ACT_DIMS[cm],
        hidden_size=64,
        batch_first=True
    ).to(device)
    obs_decoder   = MLP(input=64, hidden=64, output=6).to(device)

    wm_optimizer = torch.optim.Adam(
        list(obs_encoder.parameters()) +
        list(sequence_model.parameters()) +
        list(obs_decoder.parameters()),
        lr=1e-3
    )

    # =================================================
    # TRAINING LOOP
    # =================================================

    for epoch in range(epochs):

        traj_gen.reset()

        env_params = env_randomization(cfg, num_envs, device)
        drone.randomize_parameters(env_params)

        states = torch.zeros((num_envs, 6), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0

        epoch_policy_loss = 0.0

        for chunk_start in range(0, steps, horizon):

            chunk_states  = []
            chunk_actions = []
            chunk_pos_ref = []
            chunk_vel_ref = []

            # ------------------------------------------
            # 1️⃣ Real Environment Rollout (Policy)
            # ------------------------------------------
            for t in range(chunk_start, chunk_start + horizon):

                x, y, vx, vy, theta, omega = states.unbind(dim=1)
                pos_ref, vel_ref, acc_ref  = traj_gen.get_target(t * dt)

                obs = torch.stack([
                    pos_ref[:,0] - x,
                    pos_ref[:,1] - y,
                    vel_ref[:,0] - vx,
                    vel_ref[:,1] - vy,
                    acc_ref[:,0],
                    acc_ref[:,1],
                    torch.sin(theta),
                    torch.cos(theta),
                    omega
                ], dim=1)

                actions = policy(obs)
                states  = drone.step(states, actions, control_mode=cm)

                chunk_states.append(states)
                chunk_actions.append(actions)
                chunk_pos_ref.append(pos_ref)
                chunk_vel_ref.append(vel_ref)

            traj_chunk    = torch.stack(chunk_states)   # (T,B,6)
            actions_chunk = torch.stack(chunk_actions)

            # ------------------------------------------
            # 2️⃣ Policy Loss (Tracking)
            # ------------------------------------------
            target_pos = torch.stack(chunk_pos_ref).unsqueeze(1)
            target_vel = torch.stack(chunk_vel_ref).unsqueeze(1)

            pos_error = ((traj_chunk[..., :2] - target_pos)**2).sum(dim=1).mean()
            vel_error = ((traj_chunk[..., 2:4] - target_vel)**2).sum(dim=1).mean()
            rate_penalty = (traj_chunk[..., 5]**2).mean()

            policy_loss = pos_error + vel_error + 0.25 * rate_penalty

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            states = states.detach()
            epoch_policy_loss += policy_loss.item()

            # ------------------------------------------
            # 3️⃣ World Model Training
            # ------------------------------------------
            traj_chunk    = traj_chunk.detach()
            actions_chunk = actions_chunk.detach()

            T, B, _ = traj_chunk.shape

            wm_optimizer.zero_grad()

            # Encode states
            e_seq = obs_encoder(traj_chunk[:-1].reshape(-1, 6))
            e_seq = e_seq.reshape(T-1, B, 32)

            a_seq = actions_chunk[:-1]

            # Build GRU input
            inp_seq = torch.cat([e_seq, a_seq], dim=2)
            inp_seq = inp_seq.permute(1,0,2)   # (B, T-1, features)

            h0 = torch.zeros(1, B, 64, device=device)
            out_seq, _ = sequence_model(inp_seq, h0)

            o_hat_seq = obs_decoder(out_seq)
            target    = traj_chunk[1:].permute(1,0,2)

            wm_loss = ((o_hat_seq - target)**2).mean()

            wm_loss.backward()
            wm_optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Policy Loss: {epoch_policy_loss:.4f} | World Model Loss: {wm_loss:.6f}")

    print("Training complete.")


    # -------------------------------------------------
    # Plot o_hat vs o_target for one environment
    # -------------------------------------------------
    # if last_traj_chunk is not None and last_actions_chunk is not None:
        # with torch.no_grad():
        #     T = last_traj_chunk.shape[0]
        #     o_input = last_traj_chunk[0].to(device)
        #     h = torch.zeros(1, 1, 64, device=device)

        #     o_hat_seq = []
        #     o_target_seq = []

        #     for t in range(T - 1):
        #         a_t = last_actions_chunk[t].to(device)
        #         e_t = obs_encoder(o_input.unsqueeze(0))
        #         inp = torch.cat([e_t, a_t.unsqueeze(0)], dim=1).unsqueeze(1)
        #         out, h = sequence_model(inp, h)
        #         delta_o = obs_decoder(out.squeeze(1))
        #         o_hat = o_input + delta_o.squeeze(0)

        #         o_hat_seq.append(o_hat.cpu())
        #         o_target_seq.append(last_traj_chunk[t + 1])

        #         o_input = o_hat

        #     o_hat_seq = torch.stack(o_hat_seq).numpy()
        #     o_target_seq = torch.stack(o_target_seq).numpy()

        # fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        # dims = ["x", "y", "vx", "vy", "theta", "omega"]
        # for i, ax in enumerate(axes.flatten()):
        #     ax.plot(o_hat_seq[:, i], label="o_hat")
        #     ax.plot(o_target_seq[:, i], label="o_target")
        #     ax.set_title(dims[i])
        #     ax.grid(True, alpha=0.3)

        # axes[0, 0].legend(loc="upper right")
        # fig.suptitle("World Model: o_hat vs o_target (env 0)")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # plot_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
        # os.makedirs(plot_dir, exist_ok=True)
        # plot_path = os.path.join(plot_dir, "o_hat_vs_o_target.png")
        # fig.savefig(plot_path, dpi=150)
        # plt.close(fig)
        # print(f"Saved plot: {plot_path}")

    # -------------------------------------------------
    # Save models
    # -------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    torch.save(obs_encoder.state_dict(), os.path.join(output_dir, "obs_encoder.pt"))
    torch.save(sequence_model.state_dict(), os.path.join(output_dir, "sequence_model.pt"))
    torch.save(obs_decoder.state_dict(), os.path.join(output_dir, "obs_decoder.pt"))
    print("Policy and World model saved!\n")


        

@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def main(cfg: DictConfig):
    cm = "srt" 
    start_time = time.time()
    train(cm=cm, cfg=cfg)
    end_time = time.time()
    training_duration = end_time - start_time

    print("\n=== Training Times ===")
    print(f"Phase 1 training time: {int(training_duration // 3600)}h {int((training_duration % 3600) // 60)}m {training_duration % 60:.2f}s ")

    

if __name__ == "__main__":

    ACT_DIMS = {
        "srt": 2,
        "ctbr": 2,
        "lv": 5
    }
    
    main()