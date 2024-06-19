import numpy as np
from numpy.typing import NDArray
from numpy import dtype
import torch
#
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from typing import Any, Dict, List, Type, Union


def wrap_to_pi(angles):
    """
    Wraps *angles* to **[-pi, pi]**
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi


def softmax(input, dim=None):
    """
    Applies a softmax function using NumPy.
    Args:
        input (numpy.ndarray): input array
        dim (int): A dimension along which softmax will be computed.
    Returns:
        numpy.ndarray: softmax applied array
    """
    # if dim is None, we assume the softmax should be applied to the last dimension
    if dim is None:
        dim = -1

    # Shift input for numerical stability
    input_shifted = input - np.max(input, axis=dim, keepdims=True)
    # Calculate the exponential of the input
    exp_input = np.exp(input_shifted)
    # Sum of exponentials along the specified dimension
    sum_exp_input = np.sum(exp_input, axis=dim, keepdims=True)
    # Compute the softmax
    softmax_output = exp_input / sum_exp_input

    return softmax_output


def compute_actions_and_probs(
        policy: Policy,
        obs: Dict[str, NDArray[dtype]],
        num_agents_: int,
        explore: bool = True,
        batch_mode: bool = False,
):
    """
    Compute actions using the given policy.
    Args:
        policy (Policy): The policy to use for computing the actions.
        obs: dict
        explore (bool): Whether to use exploration when computing the actions.
    Returns:
    """
    if batch_mode:
        assert isinstance(obs, list), "When batch_mode is True, obs must be a list of observations."
        batch_size = len(obs)
        input_dict = batch_observations(obs, get_input_dict=True)
        action_info = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=explore)
        action_ = action_info[0]  # (batch_size, num_agents, num_agents)
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(batch_size, num_agents_, num_agents_, 2)  # (batch_size, num_agents, num_agents, 2)
        action_probs_ = softmax(logits, dim=-1)[:, :, :, 1]  # (batch_size, num_agents, num_agents)
    else:
        # Compute the action
        action_info = policy.compute_single_action(obs, explore=explore)
        action_ = action_info[0]
        logits = action_info[2]['action_dist_inputs']  # flattened logits (batch_size, num_agents * num_agents * 2)
        logits = logits.reshape(num_agents_, num_agents_, 2)  # (num_agents, num_agents, 2)
        action_probs_ = softmax(logits)[:, :, 1]  # (num_agents, num_agents)

    return action_, action_probs_


def batch_observations(
        obs_list: List[Dict[str, np.ndarray]],
        get_input_dict: bool = True,
        use_torch: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Batch a list of single observations into a format suitable for RLlib's compute_actions_from_input_dict method.
    Args:
        obs_list (List[Dict[str, np.ndarray]]): A list where each element is a single observation from the environment.
        get_input_dict (bool): A flag indicating whether to return the batched observations as a SampleBatch object.
        use_torch (bool): A flag indicating whether to use PyTorch tensors or numpy arrays for the batched observations.
    Returns:

    Use something like this:
    batch_size = 5

    obs_list = []
    env_list = []
    for seed in range(batch_size):
        env = LazyMsgListenersEnv(config)
        env.seed(seed)
        obs = env.reset()
        obs_list.append(obs)
        env_list.append(env)
    input_dict = batch_observations(obs_list, get_input_dict=True)
    actions = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=False)[0]
    """
    batched_obs = {}

    # Iterate over each observation and batch them together
    for obs in obs_list:
        for key, value in obs.items():
            if key not in batched_obs:
                batched_obs[key] = []
            batched_obs[key].append(value)

    # Convert lists to tensors or appropriately shaped numpy arrays
    for key, value in batched_obs.items():
        if use_torch:  # Using PyTorch tensors
            batched_obs[key] = torch.tensor(value)
        else:  # Using numpy arrays
            batched_obs[key] = np.array(value)

    if get_input_dict:
        return {SampleBatch.OBS: SampleBatch(batched_obs)}
    else:
        return batched_obs
