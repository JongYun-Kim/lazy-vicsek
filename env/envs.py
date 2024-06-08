import gym  # gym 0.23.1
from gym.utils import seeding
from gym.spaces import Box, Discrete, Dict, MultiDiscrete, MultiBinary
from copy import deepcopy
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from ray.rllib.utils.typing import (
    AgentID,
    # EnvCreator,
    # EnvID,
    # EnvType,
    MultiAgentDict,
    # MultiEnvDict,
)
from ray.tune.logger import pretty_print
from utils.my_utils_0 import wrap_to_pi
from typing import List, Optional
from pydantic import BaseModel, field_validator, model_validator, ConfigDict, conlist, conint, confloat
import yaml

# Compatibility layer for np.bool_ and np.bool
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool


class ControlConfig(BaseModel):
    speed: float = 15.0  # Speed in m/s.
    predefined_distance: float = 60.0  # Predefined distance in meters.
    communication_decay_rate: float = 1/3  # Communication decay rate.
    cost_weight: float = 1.0  # Cost weight.
    inter_agent_strength: float = 5.0  # Inter agent strength.
    bonding_strength: float = 1.0  # Bonding strength.
    k1: float = 1.0  # K1 coefficient.
    k2: float = 3.0  # K2 coefficient.
    max_turn_rate: float = 8/15  # Maximum turn rate in rad/s.
    initial_position_bound: float = 250.0  # Initial position bound in meters.


class LazyVicsekEnvConfig(BaseModel):
    seed: Optional[int] = None
    obs_dim: int = 4
    agent_name_prefix: str = 'agent_'
    env_mode: str = 'single_env'
    action_type: str = 'binary_vector'
    num_agents_pool: conlist(conint(ge=1), min_length=1)  # Must clarify it !!
    dt: float = 0.1
    comm_range: Optional[float] = None
    max_time_steps: int = 1000
    # std_p_goal will be set dynamically
    std_p_goal: Optional[float] = None
    std_v_goal: float = 0.1
    std_p_rate_goal: float = 0.1
    std_v_rate_goal: float = 0.2
    use_fixed_episode_length: bool = False
    get_state_hist: bool = False
    get_action_hist: bool = False


class Config(BaseModel):
    control: ControlConfig
    env: LazyVicsekEnvConfig

    model_config = ConfigDict(extra='forbid')

    @field_validator('control')
    def validate_control(cls, v):
        # You can add validation logic for the ControlConfig here if needed
        return v

    @field_validator('env')
    def validate_env(cls, v):
        # You can add validation logic for the LazyVicsekEnvConfig here if needed
        return v

    @model_validator(mode='after')
    def set_dependent_defaults(cls, values):
        if values.env.std_p_goal is None:
            values.env.std_p_goal = 0.7 * values.control.predefined_distance
        return values


def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


class LazyVicsekEnv(gym.Env):
    def __init__(self, yaml_path: str):
        self.config = load_config(yaml_path)
        self._validate_config()

    def _validate_config(self):
        pass

    def get_default_config_dict(self,):
        default_config = load_config('default_env_config.yaml')
        print('-------------------DEFAULT CONFIG-------------------')
        print(pretty_print(default_config.model_dump()))
        print('----------------------------------------------------')
        return deepcopy(default_config.model_dump())

    def show_current_config(self):
        print('-------------------CURRENT CONFIG-------------------')
        print(pretty_print(self.config.model_dump()))
        print('----------------------------------------------------')

    def reset(self):
        pass

    def custom_reset(self):
        pass

    def step(self, action):
        pass

    def get_obs(self):
        pass


if __name__ == "__main__":
    env = LazyVicsekEnv('default_env_config.yaml')
    print(pretty_print(env.config.model_dump()))
    print("Paused here for demonstration")
