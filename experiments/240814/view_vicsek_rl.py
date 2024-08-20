import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data
ckp_num = '0091'
date_str = '240818'
exp_num = 1
rl_path = f'../../data/{date_str}/{date_str}_RL_ckp{ckp_num}_{exp_num}.npz'
vicsek_path = f'../../data/{date_str}/{date_str}_Vicsek_ckp{ckp_num}_{exp_num}.npz'
try:
    data_rl = np.load(rl_path)
    data_vicsek = np.load(vicsek_path)
except FileNotFoundError:
    print("File not found. Please check the checkpoint number.")
    print(f"RL path: {rl_path}")
    print(f"Vicsek path: {vicsek_path}")
    exit()

# Make dir under data folder with the name of ckp_num
sv_dir = f'../../data/{date_str}/figs/{ckp_num}_{exp_num}'
os.makedirs(sv_dir, exist_ok=True)

episodic_reward_rl = data_rl['episode_reward_rl']    # (num_seeds,)
order_parameter_rl = data_rl['alignment_hists_rl']   # (num_seeds, max_time_steps) dt = 0.1s
info_util_ratio_rl = data_rl['info_usage_hists_rl']  # (num_seeds, max_time_steps)
seeds = data_rl['seeds_rl']

episodic_reward_vicsek = data_vicsek['episode_reward_vicsek']    # (num_seeds,)
order_parameter_vicsek = data_vicsek['alignment_hists_vicsek']   # (num_seeds, max_time_steps) dt = 0.1s
info_util_ratio_vicsek = data_vicsek['info_usage_hists_vicsek']  # (num_seeds, max_time_steps)

num_seeds = episodic_reward_rl.shape[0]
max_time_steps = order_parameter_rl.shape[1]

# Calculate averages
avg_episode_reward_rl = np.mean(episodic_reward_rl)
avg_episode_reward_vicsek = np.mean(episodic_reward_vicsek)
avg_last_alignment_rl = np.mean(order_parameter_rl[:, -1])
avg_last_alignment_vicsek = np.mean(order_parameter_vicsek[:, -1])
avg_info_usage_rl = np.nanmean(info_util_ratio_rl)
avg_info_usage_vicsek = np.mean(info_util_ratio_vicsek)

# Define their colors
vs_color = 'tab:blue'  # Vicsek color
rl_color = 'tab:green'

# Define grid style
plt.rcParams.update({
    'grid.color': 'gray',        # Set grid color to gray
    'grid.linestyle': '--',      # Set grid linestyle to dashed
    'grid.linewidth': 0.6,       # Set grid linewidth
    'grid.alpha': 0.7            # Set grid transparency (alpha)
})

# Define a common width for the bars with hatch for Vicsek bars and no pattern for RL
bar_width = 0.4

# Update font size and weight
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

# 1. Average episode reward
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_episode_reward_vicsek, avg_episode_reward_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''], zorder=3)
plt.title(f'Average Episode Reward ({num_seeds} experiments)')
plt.ylabel('Average Episode Reward')
plt.grid(axis='y', zorder=0)
plt.savefig(f'{sv_dir}/{date_str}_avg_episode_reward.png')
plt.show()

# 2. Average Order Parameter at t=50s
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_last_alignment_vicsek, avg_last_alignment_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''], zorder=3)
plt.title(f'Average Order Parameter at t={max_time_steps*0.1}s ({num_seeds} experiments)')
plt.ylabel('Average Order Parameter')
plt.grid(axis='y', zorder=0)
plt.savefig(f'{sv_dir}/{date_str}_avg_order_parameter.png')
plt.show()

# 3. Average Information Utilization Ratio
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_info_usage_vicsek, avg_info_usage_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''], zorder=3)
plt.title(f'Average Information Utilization Ratio ({num_seeds} experiments)')
plt.ylabel('Average Utilization Ratio')
plt.grid(axis='y', zorder=0)
plt.savefig(f'{sv_dir}/{date_str}_avg_info_util_ratio.png')
plt.show()

# 4. Information Utilization Ratio Over Time
for seed_idx, seed_ in enumerate(seeds):
    seed = int(seed_)
    info_util_ratio_at_seed_rl = info_util_ratio_rl[seed_idx, :]
    info_util_ratio_at_seed_vicsek = info_util_ratio_vicsek[seed_idx, :]
    time = np.arange(0, len(info_util_ratio_at_seed_rl) * 0.1, 0.1)

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(time, info_util_ratio_at_seed_vicsek, label='Vicsek', color='tab:blue', linewidth=3, linestyle='--')
    plt.plot(time, info_util_ratio_at_seed_rl, label='RL', color='tab:green', linewidth=1.5)
    plt.title(f'Information Utilization Ratio Over Time (seed={seed})')
    plt.xlabel('Time (s)')
    plt.ylabel('Information Utilization Ratio')
    plt.xlim(0, max_time_steps * 0.1)
    plt.ylim(0, 1.03)
    plt.legend()
    plt.savefig(f'{sv_dir}/{date_str}_s{seed}_info_util_ratio.png')
    # plt.show()
    plt.close()

# 5. Order Parameter Over Time
for seed_idx, seed_ in enumerate(seeds):
    seed = int(seed_)
    order_param_hist_at_seed_rl = order_parameter_rl[seed_idx, :]
    order_param_hist_at_seed_vicsek = order_parameter_vicsek[seed_idx, :]
    time = np.arange(0, len(order_param_hist_at_seed_rl) * 0.1, 0.1)

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(time, order_param_hist_at_seed_vicsek, label='Vicsek', color='tab:blue', linewidth=3, linestyle='--')
    plt.plot(time, order_param_hist_at_seed_rl, label='RL', color='tab:green', linewidth=1.6)
    plt.title(f'Order Parameter Over Time (seed={seed})')
    plt.xlabel('Time (s)')
    plt.ylabel('Order Parameter')
    plt.xlim(0, max_time_steps * 0.1)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.savefig(f'{sv_dir}/{date_str}_s{seed}_order_parameter.png')
    # plt.show()
    plt.close()

# 6. Print out comparison of the averages: Vicsek vs RL
print("-" * 32)
print(f"Mean episode reward of classical Vicsek policy: {avg_episode_reward_vicsek}")
print(f"Mean episode reward of RL policy: {avg_episode_reward_rl}")
print("-" * 32)
print(f"Average order parameter of classical Vicsek policy: {avg_last_alignment_vicsek}")
print(f"Average order parameter of RL policy: {avg_last_alignment_rl}")
print("-" * 32)
print(f"Average information utilization ratio of classical Vicsek policy: {avg_info_usage_vicsek}")
print(f"Average information utilization ratio of RL policy: {avg_info_usage_rl}")
print("-" * 32)
# save the averages in a text file
with open(f'{sv_dir}/{date_str}_averages.txt', 'w') as f:
    f.write(f"Mean episode reward of classical Vicsek policy: {avg_episode_reward_vicsek}\n")
    f.write(f"Mean episode reward of RL policy: {avg_episode_reward_rl}\n")
    f.write("-" * 32 + '\n')
    f.write(f"Average order parameter of classical Vicsek policy: {avg_last_alignment_vicsek}\n")
    f.write(f"Average order parameter of RL policy: {avg_last_alignment_rl}\n")
    f.write("-" * 32 + '\n')
    f.write(f"Average information utilization ratio of classical Vicsek policy: {avg_info_usage_vicsek}\n")
    f.write(f"Average information utilization ratio of RL policy: {avg_info_usage_rl}\n")
    f.write("-" * 32 + '\n')

print("Done!")