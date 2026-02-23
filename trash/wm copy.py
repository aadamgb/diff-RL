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

    print("=== Starting Training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 1 if device.type == "cpu" else 2048

    epochs  = 600
    steps   = 300
    horizon = 50
    dt      = 0.01
    # K_rollout = 40   # multi-step imagination horizon

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
    # state_dim  = 6
    # sequence_model = nn.GRU(
    #     input_size=state_dim + ACT_DIMS[cm],
    #     hidden_size=64,
    #     batch_first=True
    # ).to(device)

    # state_head = nn.Linear(64, state_dim).to(device)

    # wm_optimizer = torch.optim.Adam(
    #     list(sequence_model.parameters()) +
    #     list(state_head.parameters()),
    #     lr=1e-3
    # )

    # =================================================
    # TRAINING LOOP
    # =================================================
    for epoch in range(epochs):

        traj_gen.reset()
        # env_params = env_randomization(cfg, num_envs, device)
        # drone.randomize_parameters(env_params)

        states = torch.zeros((num_envs, 6), device=device)
        states[:, :2] = torch.rand((num_envs, 2), device=device) * 5.0

        epoch_policy_loss = 0.0
        epoch_wm_loss     = 0.0

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

            # # ------------------------------------------
            # # 3️⃣ World Model Training (Direct GRU)
            # # ------------------------------------------
            # traj_chunk    = traj_chunk.detach()
            # actions_chunk = actions_chunk.detach()

            # T, B, _ = traj_chunk.shape
            # wm_optimizer.zero_grad()
            # wm_loss = 0.0

            # for t in range(T - K_rollout):

            #     # reset hidden for each rollout start
            #     h = torch.zeros(1, B, 64, device=device)

            #     state_input = traj_chunk[t]

            #     for k in range(K_rollout):

            #         action_input = actions_chunk[t + k]

            #         inp = torch.cat([state_input, action_input], dim=1).unsqueeze(1)

            #         out, h = sequence_model(inp, h)

            #         pred_state = state_head(out.squeeze(1))

            #         target_state = traj_chunk[t + k + 1]

            #         wm_loss += ((pred_state - target_state) ** 2).mean()
                    
            #         # feed prediction forward
            #         state_input = pred_state

            # # Debug: print predictions vs targets
            # if epoch % 10 == 0 and (t+k) == 25:
            #     print(f"Step {t+k} | pred_state: x={pred_state[0,0]:.4f}, y={pred_state[0,1]:.4f}, vx={pred_state[0,2]:.4f}, vy={pred_state[0,3]:.4f}, theta={pred_state[0,4]:.4f}, omega={pred_state[0,5]:.4f}")
            #     print(f"Step {t+k} | target_state: x={target_state[0,0]:.4f}, y={target_state[0,1]:.4f}, vx={target_state[0,2]:.4f}, vy={target_state[0,3]:.4f}, theta={target_state[0,4]:.4f}, omega={target_state[0,5]:.4f}")


            # wm_loss /= (T - K_rollout) * K_rollout

            # wm_loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     list(sequence_model.parameters()) + list(state_head.parameters()),
            #     1.0
            # )
            # wm_optimizer.step()

            # epoch_wm_loss += wm_loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Policy Loss: {epoch_policy_loss:.4f} | World Model Loss: {epoch_wm_loss:.6f}")
            # print(f"Step {t+k} | pred_state: x={pred_state[0,0]:.4f}, y={pred_state[0,1]:.4f}, vx={pred_state[0,2]:.4f}, vy={pred_state[0,3]:.4f}, theta={pred_state[0,4]:.4f}, omega={pred_state[0,5]:.4f}")
            # print(f"Step {t+k} | target_state: x={target_state[0,0]:.4f}, y={target_state[0,1]:.4f}, vx={target_state[0,2]:.4f}, vy={target_state[0,3]:.4f}, theta={target_state[0,4]:.4f}, omega={target_state[0,5]:.4f}")

    # -------------------------------------------------
    # Save models
    # -------------------------------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", cm)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy.pt"))
    # torch.save(sequence_model.state_dict(), os.path.join(output_dir, "sequence_model.pt"))
    # torch.save(state_head.state_dict(), os.path.join(output_dir, "state_head.pt"))
    print("Policy and Sequence model saved!\n")


        

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