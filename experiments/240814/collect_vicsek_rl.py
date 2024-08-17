import numpy as np
import datetime
import os
# Envs and models
from env.envs import LazyVicsekEnv, load_config, config_to_env_input
from model.model_rllib import LazyVicsekModelPPO
# RLlib from Ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog


def get_next_available_number(base_path, base_filename, extension):
    """
    Get the next available number for the file name.
    """
    num = 1
    while True:
        file_name = f'{base_path}/{base_filename}_{num}.{extension}'
        if not os.path.exists(file_name):
            return num
        else:
            print(f"File {file_name} exists. Trying next number: {num + 1}")
        num += 1


def main():

    # Model settings
    model_name = "vicsek_lazy_listener"
    ModelCatalog.register_custom_model(model_name, LazyVicsekModelPPO)

    # Policy settings
    ckp_num = '0091'
    base_path_pol = "C:/Users/Jelly/ray_results/lazyvicsek0815"
    trial_path = base_path_pol + "/PPO_vicsek_lazy_env_33000_00000_0_2024-08-15_20-33-08"
    checkpoint_path = trial_path + f"/checkpoint_00{ckp_num}/policies/default_policy"
    try:
        policy = Policy.from_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print("File not found. Please check the checkpoint number.")
        print(f"checkpoint_path: {checkpoint_path}")
        exit()
    policy.model.eval()

    default_config_path = "D:/pych_ws/lazy-vicsek/env_config_train.yaml"
    my_config = load_config(default_config_path)

    my_config.env.max_time_steps = 300
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
    env.reset()
    num_agents = env.num_agents

    num_seeds = 200
    start_seed = 0
    last_seed = num_seeds - 1
    seeds = np.arange(start_seed, last_seed + 1)  # [start_seed, start_seed+1, ..., last_seed]

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

    for seed in range(start_seed, last_seed + 1):
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

    current_date = datetime.datetime.now().strftime('%y%m%d')

    base_path = f'../../data/{current_date}'
    rl_filename_base = f'{current_date}_RL_ckp{ckp_num}'
    vicsek_filename_base = f'{current_date}_Vicsek_ckp{ckp_num}'
    os.makedirs(base_path, exist_ok=True)

    # Get the next available number for RL and Vicsek files
    rl_file_number = get_next_available_number(base_path, rl_filename_base, 'npz')
    vicsek_file_number = get_next_available_number(base_path, vicsek_filename_base, 'npz')

    # save
    np.savez(f'{base_path}/{rl_filename_base}_{rl_file_number}.npz',
             episode_reward_rl=episode_reward_rl,  # (num_seeds,)
             episode_length_rl=episode_length_rl,  # (num_seeds,)
             alignment_hists_rl=alignment_hist_rl,  # (num_seeds * max_time_steps)
             info_usage_hists_rl=info_usage_hist_rl,  # (num_seeds * max_time_steps)
             seeds_rl=seeds,
             env_config_rl=my_config.dict(),
             )
    np.savez(f'{base_path}/{vicsek_filename_base}_{vicsek_file_number}.npz',
             episode_reward_vicsek=episode_reward_vicsek,  # (num_seeds,)
             episode_length_vicsek=episode_length_vicsek,  # (num_seeds,)
             alignment_hists_vicsek=alignment_hist_vicsek,  # (num_seeds * max_time_steps)
             info_usage_hists_vicsek=info_usage_hist_vicsek,  # (num_seeds * max_time_steps)
             seeds_vicsek=seeds,
             env_config_vicsek=my_config.dict(),
             )


if __name__ == "__main__":
    main()
