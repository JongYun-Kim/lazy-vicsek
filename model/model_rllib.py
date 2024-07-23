# Everything is copy
import copy
# Please let me get out of ray rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
#
# From envs
import numpy as np
from gym.spaces import Discrete
from typing import List, Union
#
# Pytorch
import torch
import torch.nn as nn
#
# For the custom model
from model.model_torch import ActorTest, CriticTest


class LazyListenerModelPPOTest(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get actor
        # assert obs_space["centralized_agents_info"].shape[2] == 4, "d_subobs is not 4"
        # self.actor = ActorTest(obs_space["centralized_agents_info"].shape[2])
        self.actor = ActorTest(4)

        # Get critic
        # self.critic = CriticTest(obs_space["centralized_agents_info"].shape[2])
        self.critic = CriticTest(4)
        self.values = None

    def forward(
            self,
            input_dict,
            state: List[TensorType],
            seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        # Get and check the observation
        obs_dict = input_dict["obs"]

        x = self.actor(obs_dict)  # (batch_size, num_agents_max * num_agents_max * 2)
        self.values = self.critic(obs_dict)  # (batch_size, 1)

        return x, state

    def value_function(self) -> TensorType:
        value = self.values.squeeze(-1)  # (batch_size,)
        return value