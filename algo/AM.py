import torch
import os

class AM:
    def __init__(self, cfg, policy, encoder=None, adaptor=None):
        self.cfg = cfg
        self.policy = policy
        self.encoder = encoder
        self.adaptor = adaptor

    def train_control(self, env, optimizer):
        """Phase 1: Train Policy + Encoder using BPTT."""
        obs_base = env.reset()
        e = env.get_privileged_info()
        epoch_loss = 0.0

        for chunk_start in range(0, self.cfg.task.steps, self.cfg.algo.horizon):
            optimizer.zero_grad()
            z = self.encoder(e)
            
            chunk_states, chunk_p_refs, chunk_v_refs = [], [], []
            
            for _ in range(self.cfg.algo.horizon):
                # Combine base observation with latent z
                obs = torch.cat([obs_base, z], dim=1)
                actions = self.policy(obs)
                obs_base = env.step(actions)
                
                chunk_states.append(env.states)
                chunk_p_refs.append(env.pos_ref.clone())
                chunk_v_refs.append(env.vel_ref.clone())
            
            traj_chunk = torch.stack(chunk_states) # (H, N, 6)
            target_pos_chunk = torch.stack(chunk_p_refs) # (H, N, 2)
            target_vel_chunk = torch.stack(chunk_v_refs) # (H, N, 2)

            pos_error = torch.mean(torch.sum((traj_chunk[..., :2] - target_pos_chunk)**2, dim=-1))
            vel_error = torch.mean(torch.sum((traj_chunk[..., 2:4] - target_vel_chunk)**2, dim=-1))
            rate_penalty = torch.mean(traj_chunk[..., 5]**2)

            loss = 1.0 * pos_error + 1.0 * vel_error + 0.25 * rate_penalty



            loss.backward()
            optimizer.step()
            
            env.states = env.states.detach()
            obs_base = obs_base.detach()
            epoch_loss += loss.item()
        return epoch_loss

    def train_adaptation(self, env, optimizer):
        """Phase 2: Train Adaptation Module to predict z."""
        env.reset()
        e = env.get_privileged_info()
        z_true = self.encoder(e).detach()
        
        history = torch.zeros((env.num_envs, self.cfg.task.k, 6 + self.policy.act_dim), device=env.device)
        obs_base = env.get_obs()
        total_loss = 0.0

        for t in range(self.cfg.task.steps):
            obs = torch.cat([obs_base, z_true], dim=1)
            actions = self.policy(obs).detach()
            
            # History management
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = torch.cat([env.states.detach(), actions], dim=1)
            
            # Predict latent
            z_hat = self.adaptor(history)
            loss = torch.mean((z_hat - z_true) ** 2)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            obs_base = env.step(actions).detach()
            env.states = env.states.detach()
            total_loss += loss.item()
        return total_loss
    

    