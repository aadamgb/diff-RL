import torch
import numpy as np

# Omega1_cmd = self.Omega_min + (self.Omega_max - self.Omega_min) * torch.sigmoid((Omega1_cmd - self.Omega_min) / (self.Omega_max - self.Omega_min) * 6 - 3)
# Omega2_cmd = self.Omega_min + (self.Omega_max - self.Omega_min) * torch.sigmoid((Omega2_cmd - self.Omega_min) / (self.Omega_max - self.Omega_min) * 6 - 3)
import matplotlib.pyplot as plt

# Parameters
omega_min = 0.0
omega_max = 4400.0

# Create range of omega_cmd values
omega_cmd = np.linspace(omega_min, omega_max, 5000)
omega_cmd_tensor = torch.tensor(omega_cmd, dtype=torch.float32)

# Apply the transformation
omega_transformed = omega_min + (omega_max - omega_min) * torch.sigmoid(
    (omega_cmd_tensor - omega_min) / (omega_max - omega_min) *6 -3
)
# omega_transformed = torch.clamp(omega_cmd_tensor, omega_min, omega_max)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(omega_cmd, omega_transformed.numpy(), linewidth=2)
plt.xlabel('omega_cmd')
plt.ylabel('Transformed omega')
plt.title('Omega Command Transformation Function')
plt.grid(True, alpha=0.3)
plt.show()