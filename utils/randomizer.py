import torch
from omegaconf import DictConfig

def env_randomization(cfg: DictConfig, num_envs=1, device="cpu"):

    # sample scale factors per environment
    c = torch.empty(num_envs, device=device).uniform_(
        cfg.dynamics.sf.min, cfg.dynamics.sf.max
    )

    # correlated scaling
    l = c * (cfg.dynamics.arm_l.max - cfg.dynamics.arm_l.min) + cfg.dynamics.arm_l.min

    m = ((l**3 - cfg.dynamics.arm_l.min**3) /
         (cfg.dynamics.arm_l.max**3 - cfg.dynamics.arm_l.min**3)) * \
        (cfg.dynamics.mass.max - cfg.dynamics.mass.min) + cfg.dynamics.mass.min

    J = ((l**5 - cfg.dynamics.arm_l.min**5) /
         (cfg.dynamics.arm_l.max**5 - cfg.dynamics.arm_l.min**5)) * \
        (cfg.dynamics.J.max - cfg.dynamics.J.min) + cfg.dynamics.J.min

    C_Dx = ((l**2 - cfg.dynamics.arm_l.min**2) /
            (cfg.dynamics.arm_l.max**2 - cfg.dynamics.arm_l.min**2)) * \
           (cfg.dynamics.C_D.x.max - cfg.dynamics.C_D.x.min) + cfg.dynamics.C_D.x.min

    C_Dy = ((l**2 - cfg.dynamics.arm_l.min**2) /
            (cfg.dynamics.arm_l.max**2 - cfg.dynamics.arm_l.min**2)) * \
           (cfg.dynamics.C_D.y.max - cfg.dynamics.C_D.y.min) + cfg.dynamics.C_D.y.min

    k1 = cfg.dynamics.thrust_map.k1.min * (
        (cfg.dynamics.thrust_map.k1.max / cfg.dynamics.thrust_map.k1.min) ** c
    )

   
    def add_noise(x):
        return x * (
            1 + torch.empty(num_envs, device=device)
                .uniform_(-cfg.dynamics.nf, cfg.dynamics.nf)
        )
    

    return {
        "l": add_noise(l),
        "m": add_noise(m),
        "J": add_noise(J),
        "C_Dx": add_noise(C_Dx),
        "C_Dy": add_noise(C_Dy),
        "k1": add_noise(k1),
    }

if __name__ == "__main__":
    env_randomization()

# TODO: motor_tau, thrust_map, min/max(T, motor_speed, omega), motor efectivness/alloc matrix