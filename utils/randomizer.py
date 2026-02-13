import torch
from omegaconf import DictConfig

def env_randomization(cfg: DictConfig, num_envs=1, device="cpu"):

    # sample scale factors per environment
    c = torch.empty(num_envs, device=device).uniform_(
        cfg.sf.min, cfg.sf.max
    )

    # correlated scaling
    l = c * (cfg.arm_l.max - cfg.arm_l.min) + cfg.arm_l.min

    m = ((l**3 - cfg.arm_l.min**3) /
         (cfg.arm_l.max**3 - cfg.arm_l.min**3)) * \
        (cfg.mass.max - cfg.mass.min) + cfg.mass.min

    J = ((l**5 - cfg.arm_l.min**5) /
         (cfg.arm_l.max**5 - cfg.arm_l.min**5)) * \
        (cfg.J.max - cfg.J.min) + cfg.J.min

    C_Dx = ((l**2 - cfg.arm_l.min**2) /
            (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * \
           (cfg.C_D.x.max - cfg.C_D.x.min) + cfg.C_D.x.min

    C_Dy = ((l**2 - cfg.arm_l.min**2) /
            (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * \
           (cfg.C_D.y.max - cfg.C_D.y.min) + cfg.C_D.y.min

    k1 = cfg.thrust_map.k1.min * (
        (cfg.thrust_map.k1.max / cfg.thrust_map.k1.min) ** c
    )

   
    def add_noise(x):
        return x * (
            1 + torch.empty(num_envs, device=device)
                .uniform_(-cfg.nf, cfg.nf)
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