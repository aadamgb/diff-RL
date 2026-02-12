import torch
import math

class BicopterDynamics:
    """
    Differentiable 2D bicopter dynamics class.

    State: [x, y, vx, vy, theta, omega]
    Action: motor/rates/velocity
    """
    def __init__(
            self,
            dt=0.01,
            mass=1.0,
            arm_length=0.2,
            inertia=0.01,
            gravity=9.81,
            max_thrust=25.0, # (Per rotor)
            # max_torque=10.0,
            device="cpu"

    ):
        self.dt = dt
        self.m = mass
        self.l = arm_length
        self.I = inertia
        self.g = gravity

        self.T_max = max_thrust
        # self.tau_max = max_torque

        self.device = device
    
    def step(self, state, action, control_mode="srt"):
        """
        Updates the dynamics. 
        """
        x, y, vx, vy, theta, omega = torch.unbind(state, dim=-1)

        T1, T2 = self._get_control(state, action, control_mode)

        T1 = self.T_max * torch.tanh(torch.nn.functional.softplus(T1) / self.T_max)
        T2 = self.T_max * torch.tanh(torch.nn.functional.softplus(T2) / self.T_max)

        T = T1 + T2
        tau = self.l * (T2 - T1)

        ax = -torch.sin(theta) * T / self.m
        ay =  torch.cos(theta) * T / self.m - self.g
        alpha = tau / self.I

        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        omega = omega + alpha * self.dt

        x = x + vx * self.dt
        y = y + vy * self.dt
        theta = theta + omega * self.dt

        return torch.stack([x, y, vx, vy, theta, omega], dim=-1)
    
    def _get_control(self, state, action, mode):
        """
        Returns individual rotor thrusts T1, T2
        """
        # Single rotor thrusts (T1, T2) 
        if mode == "srt": 
            T1 = action[..., 0]
            T2 = action[..., 1]
            return T1, T2
        
        # Collective thrust and body rate (T, omega)
        elif mode == "ctbr":
            T_cmd = action[..., 0]
            omega_cmd = action[..., 1]

            kd = 0.5
            tau = kd * (omega_cmd - state[..., 5])
            T1 = 0.5 * (T_cmd - tau / self.l)
            T2 = 0.5 * (T_cmd + tau / self.l)
            return T1, T2
            
        # Linear velocity tracking via force-based geometric control
        elif mode == "lv":
            vx, vy = state[..., 2], state[..., 3]
            theta, omega = state[..., 4], state[..., 5]

            kv = self._bounded_gain(action[..., 2], 0.0, 10.0)
            kR = self._bounded_gain(action[..., 3], 0.0, 20.0)
            kw = self._bounded_gain(action[..., 4], 0.0, 5.0)

            ax_des = kv * (action[..., 0] - vx)
            ay_des = kv * (action[..., 1] - vy)

            fx = self.m * ax_des
            fy = self.m * (ay_des + self.g)
            f_des = torch.stack([fx, fy], dim=-1)

            b = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
            T_cmd = torch.sum(f_des * b, dim=-1)

            # Desired attitude
            f_norm = torch.norm(f_des, dim=-1) + 1e-6
            b_des = f_des / f_norm.unsqueeze(-1)
            theta_des = torch.atan2(-b_des[..., 0], b_des[..., 1])

            # SO(2) geometric attitude control
            eR = torch.sin(theta - theta_des)
            tau = self.I * (-kR * eR - kw * omega)

            T1 = 0.5 * (T_cmd - tau / self.l)
            T2 = 0.5 * (T_cmd + tau / self.l)
            return T1, T2

        else:
            raise ValueError(f"Unknown control_mode: {mode}")
        
    def _bounded_gain(self, x, k_min, k_max):
        return k_min + (k_max - k_min) * torch.sigmoid(x)

