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
    current_date = datetime.datetime.now().strftime('%y%m%d')
    start_time = datetime.datetime.now()

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

    save_state_action_hist = True
    if save_state_action_hist:
        print("You are about to save the state and action history. Are you sure?")
        user_input = input("Enter 'yes' to continue, or anything else to cancel: ")
        if user_input != 'yes':
            print("State action history not saved.")
            save_state_action_hist = False
    my_config.env.get_state_hist = save_state_action_hist
    my_config.env.get_action_hist = save_state_action_hist

    env = LazyVicsekEnv(config_to_env_input(my_config))
    env.reset()
    num_agents = env.num_agents

    num_seeds = 200
    start_seed = 200
    last_seed = start_seed + num_seeds - 1
    seeds = np.arange(start_seed, last_seed + 1)  # [start_seed, start_seed+1, ..., last_seed]

    episode_reward_rl = np.zeros(num_seeds)
    episode_length_rl = np.zeros(num_seeds)
    episode_success_mask_rl = np.zeros(num_seeds, dtype=np.bool_)  # whether episode is done before max time steps
    info_usage_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
    alignment_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
    if save_state_action_hist:
        state_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, 5), dtype=np.float32)
        network_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, num_agents), dtype=np.float32)
        action_hist_rl = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, num_agents), dtype=np.float32)
    else:
        state_hist_rl, network_hist_rl, action_hist_rl = None, None, None

    episode_reward_vicsek = np.zeros(num_seeds)
    episode_length_vicsek = np.zeros(num_seeds)
    episode_success_mask_vicsek = np.zeros(num_seeds, dtype=np.bool_)  # whether episode is done before max time steps
    info_usage_hist_vicsek = np.ones((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
    alignment_hist_vicsek = np.zeros((num_seeds, my_config.env.max_time_steps), dtype=np.float32)
    if save_state_action_hist:
        state_hist_vicsek = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, 5), dtype=np.float32)
        network_hist_vicsek = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, num_agents), dtype=np.float32)
        action_hist_vicsek = np.zeros((num_seeds, my_config.env.max_time_steps, num_agents, num_agents), dtype=np.float32)
    else:
        state_hist_vicsek, network_hist_vicsek, action_hist_vicsek = None, None, None

    for seed_idx, seed_ in enumerate(seeds):
        seed = int(seed_)
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
            info_usage_hist_rl[seed_idx, step-1] = info_usage
            obs, reward, done, info = env.step(action2)
            reward_sum += reward
            alignment_hist_rl[seed_idx, step - 1] = env.alignment_hist[step - 1]
        episode_reward_rl[seed_idx] = reward_sum
        episode_length_rl[seed_idx] = env.time_step
        episode_success_mask_rl[seed_idx] = env.time_step < env.config.env.max_time_steps
        if save_state_action_hist:
            state_hist_rl[seed_idx, :env.time_step, :, :] = env.agent_states_hist
            network_hist_rl[seed_idx, :env.time_step, :, :] = env.neighbor_masks_hist
            action_hist_rl[seed_idx, :env.time_step, :, :] = env.action_hist
        print(f"[{100*(seed_idx+1)/num_seeds:.0f}%] Seed {seed} in [{start_seed}, {last_seed}]: R: {reward_sum:.2f}, L: {env.time_step}, S/F: {episode_success_mask_rl[seed_idx]}, Info: {info_usage_hist_rl[seed_idx, :env.time_step].mean()}")

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
            alignment_hist_vicsek[seed_idx, step - 1] = env.alignment_hist[step - 1]
        episode_reward_vicsek[seed_idx] = reward_sum
        episode_length_vicsek[seed_idx] = env.time_step
        episode_success_mask_vicsek[seed_idx] = env.time_step < env.config.env.max_time_steps
        if save_state_action_hist:
            state_hist_vicsek[seed_idx, :env.time_step, :, :] = env.agent_states_hist
            network_hist_vicsek[seed_idx, :env.time_step, :, :] = env.neighbor_masks_hist
            action_hist_vicsek[seed_idx, :env.time_step, :, :] = env.action_hist
        print(f"[{100*(seed_idx+1)/num_seeds:.0f}%] Seed {seed} in [{start_seed}, {last_seed}]: R: {reward_sum:.2f}, L: {env.time_step}, S/F: {episode_success_mask_vicsek[seed_idx]}, Info: {info_usage_hist_vicsek[seed_idx, :env.time_step].mean()}")

    print("-" * 32)
    print("-" * 32)
    print(f"Mean episode reward of VS policy: {np.mean(episode_reward_vicsek):.2f}")
    print(f"Mean episode reward of RL policy: {np.mean(episode_reward_rl):.2f}")
    print("-" * 32)
    print(f"Mean last order parameter of VS policy: {alignment_hist_vicsek[:, -1].mean():.4f}")
    print(f"Mean last order parameter of RL policy: {alignment_hist_rl[:, -1].mean():.4f}")
    print("-" * 32)
    print(f"Mean information utilization ratio of VS policy: {info_usage_hist_vicsek.mean():.4f}")
    print(f"Mean information utilization ratio of RL policy: {info_usage_hist_rl.mean():.4f}")
    print("-" * 32)
    print("-" * 32)

    base_path = f'../../data/{current_date}'
    rl_filename_base = f'{current_date}_RL_ckp{ckp_num}'
    vicsek_filename_base = f'{current_date}_Vicsek_ckp{ckp_num}'
    os.makedirs(base_path, exist_ok=True)

    # Get the next available number for RL and Vicsek files
    rl_file_number = get_next_available_number(base_path, rl_filename_base, 'npz')
    vicsek_file_number = get_next_available_number(base_path, vicsek_filename_base, 'npz')

    # save
    np.savez(f'{base_path}/{rl_filename_base}_{rl_file_number}.npz',
             episode_reward_rl=episode_reward_rl,     # (num_seeds,)
             episode_length_rl=episode_length_rl,     # (num_seeds,)
             alignment_hists_rl=alignment_hist_rl,    # (num_seeds * max_time_steps)
             info_usage_hists_rl=info_usage_hist_rl,  # (num_seeds * max_time_steps)
             seeds_rl=seeds,  # (num_seeds)
             env_config_rl=my_config.dict(),
             )
    np.savez(f'{base_path}/{vicsek_filename_base}_{vicsek_file_number}.npz',
             episode_reward_vicsek=episode_reward_vicsek,     # (num_seeds,)
             episode_length_vicsek=episode_length_vicsek,     # (num_seeds,)
             alignment_hists_vicsek=alignment_hist_vicsek,    # (num_seeds * max_time_steps)
             info_usage_hists_vicsek=info_usage_hist_vicsek,  # (num_seeds * max_time_steps)
             seeds_vicsek=seeds,  # (num_seeds)
             env_config_vicsek=my_config.dict(),
             )
    print("Data saved at: \n  ", base_path)
    print(f"    RL: {rl_filename_base}_{rl_file_number}.npz")
    print(f"    Vs: {vicsek_filename_base}_{vicsek_file_number}.npz \n")
    if save_state_action_hist:
        np.savez(f'{base_path}/{rl_filename_base}_{rl_file_number}_sa_hist.npz',
                 state_hist_rl=state_hist_rl,      # (num_seeds, max_time_steps, num_agents, 5)
                 network_hist_rl=network_hist_rl,  # (num_seeds, max_time_steps, num_agents, num_agents)
                 action_hist_rl=action_hist_rl,    # (num_seeds, max_time_steps, num_agents, num_agents)
                 seeds_rl=seeds,  # (num_seeds)
                 env_config_rl=my_config.dict(),
                 )
        np.savez(f'{base_path}/{vicsek_filename_base}_{vicsek_file_number}_sa_hist.npz',
                 state_hist_vicsek=state_hist_vicsek,      # (num_seeds, max_time_steps, num_agents, 5)
                 network_hist_vicsek=network_hist_vicsek,  # (num_seeds, max_time_steps, num_agents, num_agents)
                 action_hist_vicsek=action_hist_vicsek,    # (num_seeds, max_time_steps, num_agents, num_agents)
                 seeds_vicsek=seeds,  # (num_seeds)
                 env_config_vicsek=my_config.dict(),
                 )
        print("State and action history saved at: \n  ", base_path)
        print(f"    RL: {rl_filename_base}_{rl_file_number}_sa_hist.npz")
        print(f"    Vs: {vicsek_filename_base}_{vicsek_file_number}_sa_hist.npz \n")

    print("Done at ", datetime.datetime.now())
    print("Total time: ", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
