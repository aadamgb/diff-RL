import torch
from .base_env import BicopterBaseEnv

class RaceEnv(BicopterBaseEnv):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        
        # Convert list of dicts to tensors for fast GPU access
        gate_pos = [g['position'] for g in cfg.track.gates]
        gate_look = [g['look_at'] for g in cfg.track.gates]
        
        self.gate_positions = torch.tensor(gate_pos, device=device, dtype=torch.float32)
        self.gate_orientations = torch.tensor(gate_look, device=device, dtype=torch.float32)
        self.num_gates = self.gate_positions.shape[0]

        # Tracking state: which gate index (0 to N-1) is each env currently targeting
        self.target_gate_idx = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # State references
        self.pos_ref = torch.zeros((num_envs, 2), device=device)
        self.vel_ref = torch.zeros((num_envs, 2), device=device) # Usually 0 or entry velocity
        self.look_ref = torch.zeros((num_envs, 2), device=device)

    def reset(self):
        self.randomize_physics()
        # Reset drones to near the first gate
        self.states = torch.zeros((self.num_envs, 6), device=self.device)
        self.states[:, :2] = self.gate_positions[0]
        
        # Start everyone at Gate 0
        self.target_gate_idx.zero_()
        self._update_refs()
        
        return self.get_obs()

    def _update_refs(self):
        """Updates the reference tensors based on current target_gate_idx"""
        self.pos_ref = self.gate_positions[self.target_gate_idx]
        self.look_ref = self.gate_orientations[self.target_gate_idx]

    def get_obs(self):
        x, y, vx, vy, theta, omega = self.states.unbind(dim=1)
        
        # Relative position to gate and the gate's desired orientation
        return torch.stack([
            self.pos_ref[:, 0] - x, 
            self.pos_ref[:, 1] - y,
            vx, vy,
            self.look_ref[:, 0], 
            self.look_ref[:, 1],
            torch.sin(theta), 
            torch.cos(theta), 
            omega
        ], dim=1)

    def step(self, actions):
        # 1. Physics Step
        super().step(actions)
        
        # 2. Check for Gate Pass (Collision/Proximity Detection)
        # Distance to the center of the current target gate
        dist_to_gate = torch.norm(self.states[:, :2] - self.pos_ref, dim=1)
        
        # Threshold (e.g., 1.5m) to consider a gate "passed"
        passed = dist_to_gate < 1.5
        
        if passed.any():
            # Increment gate index, wrap around to 0 if track is a loop
            self.target_gate_idx[passed] = (self.target_gate_idx[passed] + 1) % self.num_gates
            self._update_refs()
            
        return self.get_obs()