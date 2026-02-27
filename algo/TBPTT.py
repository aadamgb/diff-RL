import torch
from .base_algo import BaseAlgo

class TBPTT(BaseAlgo):
    def __init__(self, policy, optimizer, cfg):
        super().__init__(policy, optimizer, cfg)

    def train_epoch(self, env):
        obs = env.reset()
        epoch_loss = 0.0
        
        buffer_states = []
        buffer_refs = []

        for t in range(self.cfg.task.steps):
            # 1. Forward Pass
            actions = self.policy(obs)
            obs = env.step(actions)
            
            # 2. Store data for backprop through time
            buffer_states.append(env.states)
            buffer_refs.append(env.pos_ref.clone())

            # 3. Horizon Update
            if (t + 1) % self.cfg.algo.horizon == 0:
                loss = self._compute_loss_position(buffer_states, buffer_refs)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Detach to prevent gradients from flowing into the next horizon
                env.states = env.states.detach()
                obs = obs.detach()
                
                epoch_loss += loss.item()
                buffer_states, buffer_refs = [], []
                
        return epoch_loss

    def _compute_loss_position(self, states, refs):
        traj_chunk = torch.stack(states)
        ref_chunk = torch.stack(refs)
        
        pos_error = torch.mean((traj_chunk[..., :2] - ref_chunk)**2)
        vel_penalty = torch.mean(traj_chunk[..., 2:4]**2)
        rate_penalty = torch.mean(traj_chunk[..., 5]**2)
        
        return 1.0 * pos_error + 0.25 * vel_penalty + 0.1 * rate_penalty
    