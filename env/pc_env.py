import torch
from .base_env import BicopterBaseEnv
from utils.rand_traj_gen import RandomTrajectoryGenerator

class BicopterHoverEnv(BicopterBaseEnv):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
        
        # Task specific states
        self.pos_ref = None
        self.vel_ref = None
        self.acc_ref = None
        self.timer = torch.zeros(num_envs, device=device)

    def reset(self):
        self.randomize_physics()
        # Initial drone state
        self.states = torch.zeros((self.num_envs, 6), device=self.device)
        self.states[:, :2] = torch.rand((self.num_envs, 2), device=self.device) * 5.0
        
        # Initial Targets
        self.pos_ref, self.vel_ref, self.acc_ref = self.traj_gen.get_hover_targets(boundary=5.0)
        self.timer.zero_()
        return self.get_obs()

    def get_obs(self):
        x, y, vx, vy, theta, omega = self.states.unbind(dim=1)
        return torch.stack([
            self.pos_ref[:, 0]-x, self.pos_ref[:, 1]-y, 
            self.vel_ref[:, 0]-vx, self.vel_ref[:, 1]-vy, 
            self.acc_ref[:, 0], self.acc_ref[:, 1], 
            torch.sin(theta), torch.cos(theta), omega
        ], dim=1)

    def step(self, actions):
        # 1. Physics Step
        super().step(actions)
        
        # 2. Update Timer and check for target resets
        dist = torch.sqrt(((self.pos_ref - self.states[:, :2])**2).sum(dim=1))
        close_envs = dist < 0.5 
        self.timer[close_envs] += self.cfg.dt
        self.timer[~close_envs] = 0.0

        reset_pos = self.timer > 3.0 
        if reset_pos.any():
            new_p, new_v, new_a = self.traj_gen.get_hover_targets(boundary=5.0)
            self.pos_ref[reset_pos] = new_p[reset_pos]
            self.vel_ref[reset_pos] = new_v[reset_pos]
            self.acc_ref[reset_pos] = new_a[reset_pos]
            self.timer[reset_pos] = 0.0
            
        return self.get_obs()