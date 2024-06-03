import gym  # gym 0.23.1
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict, MultiDiscrete, MultiBinary
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ray.rllib.utils.typing import (
    AgentID,
    # EnvCreator,
    # EnvID,
    # EnvType,
    MultiAgentDict,
    # MultiEnvDict,
)

matplotlib.use('TkAgg')  # To avoid the MacOS backend; but just follow your needs

# Compatibility layer for np.bool_ and np.bool
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool

class LazyVicsekEnv(gym.Env):

    def __init__(self, config):
        pass

    def _validate_config(self):
        pass

    def get_default_config(self, extra_config):
        pass

    def reset(self):
        pass

    def custom_reset(self):
        pass

    def step(self, action):
        pass

    def get_obs(self):
        pass


