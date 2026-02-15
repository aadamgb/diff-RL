import torch
import numpy as np
from omegaconf import DictConfig

#    T1       T2
#  _____    _____
#    |________|

class BicopterDynamics:
    """
    Differentiable 2D bicopter dynamics class.

    State: [x, y, vx, vy, theta, omega, Omega1, Omega2]
    """
    def __init__(self, cfg: DictConfig, dt=0.01, device="cpu"):

        self.dt = dt
        self.device = device
        self.cfg = cfg

        self.l = torch.tensor(cfg.arm_l.nominal, device=device)
        self.m = torch.tensor(cfg.mass.nominal, device=device)
        self.J = torch.tensor(cfg.J.nominal, device=device)

        self.g = torch.tensor(cfg.g, device=device)

        self.C_Dx = torch.tensor(cfg.C_D.x.nominal, device=device)
        self.C_Dy = torch.tensor(cfg.C_D.y.nominal, device=device)
        self.rho = torch.tensor(cfg.rho, device=device)

        self.k1 = torch.tensor(cfg.thrust_map.k1.nominal, device=device)

        self.km_up = torch.tensor(cfg.km.up.nominal, device=device)
        self.km_down = torch.tensor(cfg.km.down.nominal, device=device)

        self.Omega_min = torch.tensor(cfg.motor_speed_min, device=device)
        self.Omega_max = torch.tensor(cfg.motor_speed_max, device=device)
        
        self.Omega_dot_min = torch.tensor(cfg.motor_acc_min, device=device)
        self.Omega_dot_max = torch.tensor(cfg.motor_acc_max, device=device)

        self.Ti_max = torch.tensor(cfg.Ti_max, device=device)

        self.eps = 1e-4

    
    def step(self, state, action, control_mode="srt"):
        """
        Updates the dynamics. 
        """
        x, y, vx, vy, theta, omega = torch.unbind(state, dim=-1)

        T1_cmd, T2_cmd = self._get_control(state, action, control_mode)

        # Softplus and saturate commanded thrust
        T1 = self.Ti_max * torch.tanh(torch.nn.functional.softplus(T1_cmd) / self.Ti_max)
        T2 = self.Ti_max * torch.tanh(torch.nn.functional.softplus(T2_cmd) / self.Ti_max)

        T = T1 + T2
        tau = self.l * (T2 - T1)

        # Calculate drag forces
        drag_x, drag_y = self._calculate_drag(vx, vy, theta)

        # Translational dynamics
        ax =  (-torch.sin(theta) * T - drag_x) / self.m
        ay =  ( torch.cos(theta) * T - self.m * self.g - drag_y) / self.m
        
        # Rotational dynamics
        alpha = tau / self.J

        # Integrate (Semi-Implicit Euler)
        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        omega = omega + alpha * self.dt

        x = x + vx * self.dt
        y = y + vy * self.dt
        theta = theta + omega * self.dt
        return torch.stack([x, y, vx, vy, theta, omega], dim=-1)
    
    def _calculate_drag(self, vx, vy, theta):
        """
        Calculate translational drag forces using quadratic drag model in body frame.
        Converts world-frame velocities to body frame, calculates drag, then converts back to world frame.
        (note that rotational body drag and propeller drag is neglected)
        """
        # Convert world-frame velocities to body-frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        vx_body = vx * cos_theta + vy * sin_theta
        vy_body = -vx * sin_theta + vy * cos_theta
        
        # Calculate drag in body frame (quadratic drag)
        v_norm = torch.sqrt(vx_body**2 + vy_body**2 + 1e-6)
        area_x = self.l * 0.1       # 0.1 arbitrary area sacle down factor
        area_y = self.l * self.l    # adding fake depth  
        drag_x_body = 0.5 * self.rho * self.C_Dx * area_x * v_norm * vx_body
        drag_y_body = 0.5 * self.rho * self.C_Dy * area_y * v_norm * vy_body
        
        # Convert body-frame drag forces back to world-frame
        drag_x = drag_x_body * cos_theta - drag_y_body * sin_theta
        drag_y = drag_x_body * sin_theta + drag_y_body * cos_theta
        
        return drag_x, drag_y
    
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

            tau_cmd = self.J * (omega_cmd - state[..., 5]) / self.dt

            T1 = 0.5 * (T_cmd - tau_cmd / self.l)
            T2 = 0.5 * (T_cmd + tau_cmd / self.l)
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
            tau = self.J * (-kR * eR - kw * omega)

            T1 = 0.5 * (T_cmd - tau / self.l)
            T2 = 0.5 * (T_cmd + tau / self.l)
            return T1, T2

        else:
            raise ValueError(f"Unknown control_mode: {mode}")
        
    def _bounded_gain(self, x, k_min, k_max):
        return k_min + (k_max - k_min) * torch.sigmoid(x)
    
    # TODO: Not sure wether this is the best way of clamping
    def _differentiable_clamp(self, x, xmin, xmax):
        return xmin + (xmax - xmin) * torch.sigmoid((x - xmin) / (xmax - xmin) * 6 - 3)

    def randomize_parameters(self, params: dict):
        '''
        Sets the randomized parameters 
        '''
        if "m" in params:
            self.m = params["m"].to(self.device)
        if "l" in params:
            self.l = params["l"].to(self.device)
        if "J" in params:
            self.J = params["J"].to(self.device)
        if "C_Dx" in params:
            self.C_Dx = params["C_Dx"].to(self.device)
        if "C_Dy" in params:
            self.C_Dy = params["C_Dy"].to(self.device)
        if "k1" in params:
            self.k1 = params["k1"].to(self.device)

    def get_env_parameters(self):
        '''
        Returns the current parameters as a dictionary
        '''
        return {
            "m": self.m,
            "l": self.l,
            "J": self.J,
            "C_Dx": self.C_Dx,
            "C_Dy": self.C_Dy,
            "k1": self.k1,
        }

    def motor_hover_speed(self):
        '''
        Returns the required motor angluar velocity to hover
        '''
        return torch.sqrt(self.m * self.g / (2 * self.k1))
        


    


