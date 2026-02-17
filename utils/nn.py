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
    
class MLP(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output)
        )

    def forward(self, x):
        return self.net(x)


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


class DeterministicRSSM(nn.Module):
    def __init__(self, obs_dim, act_dim, embed_dim=32, hidden_dim=64):
        super().__init__()

        # Observation encoder
        self.encoder = MLP(obs_dim, 64, embed_dim)

        # GRU sequence model
        self.gru = nn.GRU(
            input_size=embed_dim + act_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Decoder (predict next observation)
        self.decoder = MLP(hidden_dim, 64, obs_dim)

        self.hidden_dim = hidden_dim

    def forward(self, obs_seq, act_seq, h0=None):
        """
        obs_seq: (B, T, obs_dim)
        act_seq: (B, T, act_dim)

        Returns:
            pred_obs: (B, T, obs_dim)
            h_T: final hidden state
        """

        B, T, _ = obs_seq.shape

        # Encode observations
        obs_flat = obs_seq.reshape(B * T, -1)
        embed = self.encoder(obs_flat)
        embed = embed.reshape(B, T, -1)

        # Concatenate embedding and action
        gru_input = torch.cat([embed, act_seq], dim=-1)

        # Initialize hidden state
        if h0 is None:
            h0 = torch.zeros(1, B, self.hidden_dim, device=obs_seq.device)

        # GRU forward
        h_seq, h_T = self.gru(gru_input, h0)

        # Decode predictions
        h_flat = h_seq.reshape(B * T, -1)
        pred_obs = self.decoder(h_flat)
        pred_obs = pred_obs.reshape(B, T, -1)

        return pred_obs, h_T
