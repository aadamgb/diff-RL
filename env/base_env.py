import torch
from dynamics.bicopter_dynamics import BicopterDynamics
from utils.randomizer import env_randomization

class BicopterBaseEnv:
    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.drone = BicopterDynamics(device=device, cfg=cfg)
        
        self.states = None
        self.env_params = None

    def randomize_physics(self):
        self.env_params = env_randomization(self.cfg, self.num_envs, self.device)
        self.drone.randomize_parameters(self.env_params)

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        # Physics update is common to all tasks
        self.states = self.drone.step(self.states, actions, control_mode=self.cfg.cm)