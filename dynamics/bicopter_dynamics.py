import torch
import numpy as np
from omegaconf import DictConfig

#    T1       T2
#  _____    _____
#    |________|

class BicopterDynamics:
    """
    Differentiable 2D bicopter dynamics class.

    State: [x, y, vx, vy, theta, omega]
    Action: [Ω1, Ω2] (rotor speeds)
    """
    def __init__(self, cfg: DictConfig, dt=0.01, device="cpu"):

        self.dt = dt
        self.device = device
        self.cfg = cfg

        self.m = cfg.mass.nominal
        self.l = cfg.arm_l.nominal
        self.I = cfg.J.nominal
        self.g = cfg.g

        self.C_Dx = cfg.C_D.x.nominal
        self.C_Dy = cfg.C_D.y.nominal
        self.rho = cfg.rho

        self.k1 = cfg.thrust_map.k1.nominal
        self.km_up = cfg.km.up.nominal
        self.km_down = cfg.km.down.nominal

        self.Ti_max = cfg.Ti_max  # per rotor
    
    def step(self, state, action, control_mode="srt"):
        """
        Updates the dynamics. 
        """
        x, y, vx, vy, theta, omega, Omega1, Omega2 = torch.unbind(state, dim=-1)

        T1_cmd, T2_cmd = self._get_control(state, action, control_mode)

        T1_cmd = self.Ti_max * torch.tanh(torch.nn.functional.softplus(T1_cmd) / self.Ti_max)
        T2_cmd = self.Ti_max * torch.tanh(torch.nn.functional.softplus(T2_cmd) / self.Ti_max)

        T = T1_cmd + T2_cmd
        tau = self.l * (T2_cmd - T1_cmd)

        allocation_matrix = np.array([[1.0, 1.0], [-self.l, self.l]])

        # Drag  TODO: Maybe drag coeffs would be even worth to randomize at each time step**
        v_norm = torch.sqrt(vx**2 + vy**2 + 1e-6)
        area_x = self.l * 0.1    # 0.1 is just an arbitrary factor
        area_y = self.l
        drag_x = 0.5 * self.rho * self.C_Dx * area_x * v_norm * vx
        drag_y = 0.5 * self.rho * self.C_Dy * area_y * v_norm * vy

        # Translational dynamics
        ax =  (-torch.sin(theta) * T - drag_x) / self.m
        ay =  ( torch.cos(theta) * T - self.m * self.g - drag_y) / self.m
        
        # Rotational dynamics
        alpha = tau / self.I

        # Integrate
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
    
    def _calculate_drag(self, state):

        v_body = state[..., 2:3]
        
        

        # f_drag = -0.5 * self.rho * self.C_D * self.area * v_body.norm() * v_body

        # return f_drag
        pass

    def randomize_parameters(self, params: dict):
        '''
        Setting the randomized params 
        '''
        self.m = params["m"]
        self.l = params["l"]
        self.I = params["J"]
        self.C_Dx = params["C_Dx"]
        self.C_Dy = params["C_Dy"]
        self.k1 = params["k1"]

    


