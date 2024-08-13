import numpy as np
# Envs and models
from env.envs import LazyVicsekEnv, load_config, config_to_env_input
from model.model_rllib import LazyVicsekModelPPO
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


# Model settings
model_name = "vicsek_lazy_listener"
ModelCatalog.register_custom_model(model_name, LazyVicsekModelPPO)

# Policy settings
base_path = "C:/Users/Jelly/ray_results/debugging0812"
trial_path = base_path + "/PPO_vicsek_lazy_env_75b51_00000_0_2024-08-12_21-42-59"
checkpoint_path = trial_path + "/checkpoint_000029/policies/default_policy"
policy = Policy.from_checkpoint(checkpoint_path)
policy.model.eval()

default_config_path = "D:/pych_ws/lazy-vicsek/env_config_train.yaml"
my_config = load_config(default_config_path)

my_config.env.max_time_steps = 500
my_config.env.alignment_window_length = 32
my_config.env.alignment_goal = 0.97
my_config.env.alignment_rate_goal = 0.02
my_config.env.comm_range = 10
my_config.control.initial_position_bound = 100.0
my_config.control.speed = 5.0
my_config.env.num_agents_pool = [20]
# my_config.control.max_turn_rate = 8/15

my_config.env.use_fixed_episode_length = True
my_config.env.ignore_comm_lost_agents = True
my_config.env.periodic_boundary = True

env = LazyVicsekEnv(config_to_env_input(my_config))
obs = env.reset()
num_agents = env.num_agents

num_seeds = 2

episode_reward_rl = np.zeros(num_seeds)
episode_length_rl = np.zeros(num_seeds)
episode_success_mask_rl = np.zeros(num_seeds, dtype=np.bool_)  # whether episode is done before max time steps
info_usage_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
alignment_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)

episode_reward_vicsek = np.zeros(num_seeds)
episode_length_vicsek = np.zeros(num_seeds)
episode_success_mask_vicsek = np.zeros(num_seeds, dtype=np.bool_)  # whether episode is done before max time steps
info_usage_hist_vicsek = np.ones((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
alignment_hist_vicsek = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)

for seed in range(num_seeds):
    # RL
    done = False
    env.seed(seed)
    obs = env.reset()
    reward_sum = 0
    step = 0
    while not done:
        step += 1
        action = env.get_vicsek_action()
        action2, _, action2_info = policy.compute_single_action(obs, explore=True)  # didn't control torch seed.. my bad
        info_usage = (action2.sum()-num_agents)/(action.sum()-num_agents)
        info_usage_hist_rl[seed, step-1] = info_usage
        obs, reward, done, info = env.step(action2)
        reward_sum += reward
        alignment_hist_rl[seed, step - 1] = env.alignment_hist[step - 1]
    episode_reward_rl[seed] = reward_sum
    episode_length_rl[seed] = env.time_step
    episode_success_mask_rl[seed] = env.time_step < env.config.env.max_time_steps
    print(f"Seed {seed+1}/{num_seeds}: R: {reward_sum}, L: {env.time_step}, S/F: {episode_success_mask_rl[seed]}, Info: {info_usage_hist_rl[seed, :env.time_step].mean()}")

    # Vicsek
    done = False
    env.seed(seed)
    obs = env.reset()
    reward_sum = 0
    step = 0
    while not done:
        step += 1
        action = env.get_vicsek_action()
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        alignment_hist_vicsek[seed, step - 1] = env.alignment_hist[step - 1]
    episode_reward_vicsek[seed] = reward_sum
    episode_length_vicsek[seed] = env.time_step
    episode_success_mask_vicsek[seed] = env.time_step < env.config.env.max_time_steps
    print(f"Seed {seed+1}/{num_seeds}: R: {reward_sum}, L: {env.time_step}, S/F: {episode_success_mask_vicsek[seed]}, Info: {info_usage_hist_vicsek[seed, :env.time_step].mean()}")

# save
np.savez('../../data/240814/240814_RL.npz',
         episode_reward_rl=episode_reward_rl,  # (num_seeds,)
         episode_length_rl=episode_length_rl,  # (num_seeds,)
         alignment_hists_rl=alignment_hist_rl,  # (num_seeds * max_time_steps)
         info_usage_hists_rl=info_usage_hist_rl)  # (num_seeds * max_time_steps)
np.savez('../../data/240814/240814_Vicsek.npz',
         episode_reward_vicsek=episode_reward_vicsek,  # (num_seeds,)
         episode_length_vicsek=episode_length_vicsek,  # (num_seeds,)
         alignment_hists_vicsek=alignment_hist_vicsek,  # (num_seeds * max_time_steps)
         info_usage_hists_vicsek=info_usage_hist_vicsek)  # (num_seeds * max_time_steps)
