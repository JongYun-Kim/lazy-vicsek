import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_rl = np.load('../../data/240814/240814_RL.npz')
data_vicsek = np.load('../../data/240814/240814_Vicsek.npz')

# Extract the required data
episode_reward_rl = data_rl['episode_reward_RL']  # (40,)
last_alignments_rl = data_rl['last_alignments_RL']  # (40,)
info_usage_rl = data_rl['info_usage_hist_RL']  # (40, 500)
alignment_hist_40_rl = data_rl['alignment_hist_seed_40_RL']  # (500,)
#
episode_reward_vicsek = data_vicsek['episode_reward_vicsek']
last_alignments_vicsek = data_vicsek['last_alignments_vicsek']
info_usage_vicsek = data_vicsek['info_usage_hist_vicsek']
alignment_hist_40_vicsek = data_vicsek['alignment_hist_seed_40_vicsek']

# Calculate averages
avg_episode_reward_rl = np.mean(episode_reward_rl)
avg_episode_reward_vicsek = np.mean(episode_reward_vicsek)
avg_last_alignment_rl = np.mean(last_alignments_rl)
avg_last_alignment_vicsek = np.mean(last_alignments_vicsek)
avg_info_usage_rl = np.mean(info_usage_rl)
avg_info_usage_vicsek = np.mean(info_usage_vicsek)

# Define a common width for the bars with hatch for Vicsek bars and no pattern for RL
bar_width = 0.4

# Update font size and weight
plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})

# 1. Average episode reward with hatch pattern for Vicsek and no pattern for RL
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_episode_reward_vicsek, avg_episode_reward_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''])
plt.title('Average Episode Reward')
plt.ylabel('Average Reward')
plt.show()

# 2. Replace "last alignment" with "order parameter at t=50s"
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_last_alignment_vicsek, avg_last_alignment_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''])
plt.title('Average Order Parameter at t=50s')
plt.ylabel('Average Order Parameter')
plt.show()

# 3. Average information usage with hatch pattern for Vicsek and no pattern for RL
plt.figure(figsize=(8, 6))
plt.bar(['Vicsek', 'RL'], [avg_info_usage_vicsek, avg_info_usage_rl], color=['tab:blue', 'tab:green'], width=bar_width, hatch=['/', ''])
plt.title('Average Information Usage')
plt.ylabel('Average Usage')
plt.show()

# 4. Replace the title to "Information Usage Over Time (seed=40)"
# Correct time dimension
info_usage_rl_seed40 = info_usage_rl[39, :]
info_usage_vicsek_seed40 = info_usage_vicsek[39, :]
time = np.arange(0, len(info_usage_rl_seed40) * 0.1, 0.1)

plt.figure(figsize=(10, 6))
plt.plot(time, info_usage_vicsek_seed40, label='Vicsek', color='tab:blue', linewidth=3, linestyle='--')
plt.plot(time, info_usage_rl_seed40, label='RL', color='tab:green', linewidth=2)
plt.title('Information Usage Over Time (seed=40)')
plt.xlabel('Time (seconds)')
plt.ylabel('Information Usage')
plt.legend()
plt.show()

# 5. Replace the title and y-label in the alignment plot
plt.figure(figsize=(10, 6))
plt.plot(time, alignment_hist_40_vicsek, label='Vicsek', color='tab:blue', linewidth=3, linestyle='--')
plt.plot(time, alignment_hist_40_rl, label='RL', color='tab:green', linewidth=2)
plt.title('Order Parameter Over Time (seed=40)')
plt.xlabel('Time (seconds)')
plt.ylabel('Order Parameter')
plt.legend()
plt.show()
