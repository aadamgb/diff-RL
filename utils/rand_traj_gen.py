import torch

class RandomTrajectoryGenerator:
    def __init__(self, num_envs, device, num_harmonics=5):
        self.num_envs = num_envs
        self.device = device
        self.num_harmonics = num_harmonics
        self.reset()

    def reset(self):
        """Generates new random coefficients for each environment."""
        # Random amplitudes: (num_envs, 2 dimensions, num_harmonics)
        self.amps = (torch.rand((self.num_envs, 2, self.num_harmonics), device=self.device) * 1.0) + 0.5
        
        # Random frequencies: (num_envs, 2, num_harmonics)
        self.freqs = (torch.rand((self.num_envs, 2, self.num_harmonics), device=self.device) * 2.0) + 0.2
        
        # Random phases: (num_envs, 2, num_harmonics)
        self.phases = torch.rand((self.num_envs, 2, self.num_harmonics), device=self.device) * 2 * torch.pi

    def get_target(self, t_float):
        """
        Calculates target position and velocity for time t.
        Formula: x(t) = sum( A * sin(w*t + phi) )
        """
        # t_float is a scalar or a tensor of shape (num_envs, 1)
        # Reshape for broadcasting: (num_envs, 1, 1)
        t = torch.full((self.num_envs, 1, 1), t_float, device=self.device)

        # Position: sum across the harmonics dimension (dim=2)
        # Result shape: (num_envs, 2) -> [x, y]
        pos = torch.sum(self.amps * torch.sin(self.freqs * t + self.phases), dim=2)

        # Velocity: d/dt sum( A * sin(w*t + phi) ) = sum( A * w * cos(w*t + phi) )
        vel = torch.sum(self.amps * self.freqs * torch.cos(self.freqs * t + self.phases), dim=2)

        # Acceleration: d/dt sum( A * w * cos(w*t + phi) ) = sum( -A * w^2 * sin(w*t + phi) )
        acc = torch.sum(-self.amps * (self.freqs ** 2) * torch.sin(self.freqs * t + self.phases), dim=2)

        return pos.float(), vel.float(), acc.float()