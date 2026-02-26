import hydra
from omegaconf import DictConfig
import importlib

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        task_module = importlib.import_module(f"task.{cfg.task.name}")
    except ImportError:
        print(f"Error: Task '{cfg.task.name}' not found in task/ folder.")
        return

    print(f"Executing Environment: {cfg.task.name.upper()} with Control Mode: {cfg.cm.upper()} and Algorithm: {cfg.algo.name} ")
    
    task_module.train(cfg)

if __name__ == "__main__":
    main()