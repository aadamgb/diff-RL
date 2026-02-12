import torch
import torch.nn as nn

class BicopterPolicy(nn.Module):
    def __init__(self, obs_dim=9, act_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs):
        """
        returns actions: (N, 2)
        """
        return self.net(obs)
