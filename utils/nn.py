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



class IntrinsicsEncoder(nn.Module):
    def __init__(self, e_dim=5, z_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, z_dim)
        )

    def forward(self, e):
        return self.net(e)


class AdaptationModule(nn.Module):
    def __init__(self, input_dim=8, z_dim=2, k=20):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * k, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim)
        )

    def forward(self, history):
        # history: (B, k, 8)
        x = history.permute(0, 2, 1)  # (B, 8, k)
        x = self.conv(x)              # (B, 32, k)
        x = x.flatten(1)              # (B, 32*k)
        z_hat = self.fc(x)
        return z_hat
