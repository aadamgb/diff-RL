import hydra
import numpy as np
from omegaconf import DictConfig

def scale_randomization(cfg: DictConfig):
    c = np.round(np.random.uniform(cfg.sf.min, cfg.sf.max), 3)
    l = np.round(c * (cfg.arm_l.max - cfg.arm_l.min) + cfg.arm_l.min, 3)
    m = np.round(((l**3 - cfg.arm_l.min**3) / (cfg.arm_l.max**3 - cfg.arm_l.min**3)) * (cfg.mass.max - cfg.mass.min) + cfg.mass.min, 3)
    J = np.round(((l**5 - cfg.arm_l.min**5) / (cfg.arm_l.max**5 - cfg.arm_l.min**5)) * (cfg.J.max - cfg.J.min) + cfg.J.min, 3)
    C_Dx = np.round(((l**2 - cfg.arm_l.min**2) / (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * (cfg.C_D.x.max - cfg.C_D.x.min) + cfg.C_D.x.min, 3)
    C_Dy = np.round(((l**2 - cfg.arm_l.min**2) / (cfg.arm_l.max**2 - cfg.arm_l.min**2)) * (cfg.C_D.y.max - cfg.C_D.y.min) + cfg.C_D.y.min, 3)
    C_F = np.round(cfg.C_F.min * (cfg.C_F.max / cfg.C_F.min)**c, 3)

    print(f"\nScale factor c = {c} | Noise factor ={cfg.nf}\n")
    
    return {"l": l, "m": m, "J": J, "C_Dx": C_Dx, "C_Dy": C_Dy, "C_F": C_F}

@hydra.main(config_path="../cfg/dynamics", config_name="bicopter", version_base=None)
def noise_randomization(cfg: DictConfig):
    params = scale_randomization(cfg)
    noisy_params = {k: np.round(v * (1 + np.random.uniform(-cfg.nf, cfg.nf)), 3) 
                    for k, v in params.items()}
    
    for key in params:
        print(f"{key}: {params[key]} (scaled) -> {noisy_params[key]} (noisy)")
    
    return noisy_params

if __name__ == "__main__":
    noise_randomization()

# TODO: motor_tau, thrust_map, min/max(T, motor_speed, omega), motor efectivness/alloc matrix