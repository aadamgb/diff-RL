import torch
from .base_env import BicopterBaseEnv
from utils.rand_traj_gen import RandomTrajectoryGenerator

class TrackingEnv(BicopterBaseEnv):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.traj_gen = RandomTrajectoryGenerator(num_envs=num_envs, device=device)
        self.t_step = 0

        self.pos_ref = None
        self.vel_ref = None
        self.acc_ref = None

    def reset(self):
        self.randomize_physics()
        self.traj_gen.reset()
        self.t_step = 0
        self.states = torch.zeros((self.num_envs, 6), device=self.device)
        # Random initial pos
        self.states[:, :2] = torch.rand((self.num_envs, 2), device=self.device) * 5.0
        
        # Initialize the references
        self.pos_ref, self.vel_ref, self.acc_ref = self.traj_gen.get_target(0.0)
        return self.get_obs()

    def get_obs(self):
        x, y, vx, vy, theta, omega = self.states.unbind(dim=1)
        
        return torch.stack([
            self.pos_ref[:, 0] - x, self.pos_ref[:, 1] - y, 
            self.vel_ref[:, 0] - vx, self.vel_ref[:, 1] - vy, 
            self.acc_ref[:, 0], self.acc_ref[:, 1], 
            torch.sin(theta), torch.cos(theta), omega
        ], dim=1)

    def step(self, actions):
        self.states = self.drone.step(self.states, actions, control_mode=self.cfg.cm)
        self.t_step += 1
        
        self.pos_ref, self.vel_ref, self.acc_ref = self.traj_gen.get_target(self.t_step * self.cfg.dt)
        
        return self.get_obs()
    

    def get_privileged_info(self):
        """Returns the 'e' vector containing actual physics parameters."""
        p = self.env_params
        return torch.stack([p["m"], p["J"], p["l"], p["C_Dx"], p["C_Dy"]], dim=1)