import hydra
from omegaconf import DictConfig
import importlib

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        env_module = importlib.import_module(f"env.{cfg.env.name}")
    except ImportError:
        print(f"Error: Environment '{cfg.env.name}' not found in env/ folder.")
        return

    print(f"Executing Environment: {cfg.env.abbr.upper()} with Control Mode: {cfg.cm.upper()}")
    
    env_module.train(cfg)

if __name__ == "__main__":
    main()