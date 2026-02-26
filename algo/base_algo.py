class BaseAlgo:
    def __init__(self, policy, optimizer, cfg):
        self.policy = policy
        self.optimizer = optimizer
        self.cfg = cfg

    def update(self, env):
        raise NotImplementedError