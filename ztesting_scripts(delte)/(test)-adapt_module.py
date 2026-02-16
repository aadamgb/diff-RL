import torch
import hydra
from utils.nn import IntrinsicsEncoder
from utils.randomizer import env_randomization
from omegaconf import DictConfig


@hydra.main(config_path="cfg/dynamics", config_name="bicopter", version_base=None)
def encode(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    num_envs = 1 if device == "cpu" else  8

    env_params = env_randomization(cfg, num_envs, device)
    m = env_params["m"].to(device)
    J = env_params["J"].to(device)
    l = env_params["l"].to(device)
    C_Dx = env_params["C_Dx"].to(device)
    C_Dy = env_params["C_Dy"].to(device)

    e = torch.stack([m, J, l, C_Dx, C_Dy], dim=1)

    print(e)


    env_encoder = IntrinsicsEncoder(e_dim=5, z_dim=2).to(device)

    z = env_encoder(e)

    print(z)

if __name__ == "__main__":
    encode()