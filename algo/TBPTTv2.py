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
        buffer_looks = []

        for t in range(self.cfg.task.steps):
            # 1. Forward Pass
            actions = self.policy(obs)
            obs = env.step(actions)
            
            # 2. Store data for backprop through time
            buffer_states.append(env.states)
            buffer_refs.append(env.pos_ref.clone())
            buffer_looks.append(env.look_ref.clone()) # Track orientations!

            # 3. Horizon Update
            if (t + 1) % self.cfg.algo.horizon == 0:
                loss = self._compute_loss_racing(buffer_states, buffer_refs, buffer_looks)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Detach to prevent gradients from flowing into the next horizon
                env.states = env.states.detach()
                obs = obs.detach()
                
                epoch_loss += loss.item()
                buffer_states, buffer_refs, buffer_looks = [], [], []
                
        return epoch_loss

    def _compute_loss_racing(self, states, refs, look_refs):
        traj_chunk = torch.stack(states)   # [Horizon, Num_Envs, 6]
        ref_chunk = torch.stack(refs)      # [Horizon, Num_Envs, 2]
        look_chunk = torch.stack(look_refs) # [Horizon, Num_Envs, 2]

        # 1. Position Error (Stay close to the track)
        pos_error = torch.mean((traj_chunk[..., :2] - ref_chunk)**2)

        # 2. Alignment Loss (Look at the gate's exit direction)
        # We want the velocity vector to align with look_at
        vel = traj_chunk[..., 2:4]
        # Penalize velocity components perpendicular to the gate's look_at direction
        # This encourages "shooting" through the gate
        gate_dir = look_chunk / (torch.norm(look_chunk, dim=-1, keepdim=True) + 1e-6)

        # 2. Calculate velocity perpendicular to the gate direction
        # vel_perp = vel - (vel dot gate_dir) * gate_dir
        dot_product = (vel * gate_dir).sum(dim=-1, keepdim=True)
        vel_perp = vel - dot_product * gate_dir

        # 3. Penalize only the perpendicular velocity (straying off course)
        # and reward the dot product (speed in the right direction)
        alignment_loss = torch.mean(vel_perp**2) - 0.1 * torch.mean(dot_product)

        # 3. Control Smoothness (Minimize angular velocity omega)
        rate_penalty = torch.mean(traj_chunk[..., 5]**2)

        return 1.0 * pos_error + 0.5 * alignment_loss + 0.1 * rate_penalty