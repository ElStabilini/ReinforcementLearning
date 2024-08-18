from pathlib import Path
import gymnasium as gym
import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm

from environment_v01 import VFAFarmEnv
from algorithms import DQNAgent

# Define paths to saving directories
TrainedDQLearning_model = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Model')
TrainedDQLearning_data = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Data')
TrainedDQLearning_plots = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Plots')

# Initialize agent and environment
env = gym.make('VFAFarm-v0.0.1')
agent = DQNAgent(env)

num_episodes = 1000
batch_size = 32

# Lists to store rewards, episode lengths, and other data
rewards = []
episode_lengths = []
episode_info = []


for episode in tqdm(range(num_episodes)):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    episode_additional_info = {}

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        steps += 1

        # Accumulate info for the episode
        for key, value in info.items():
            if key not in episode_additional_info:
                episode_additional_info[key] = []
            episode_additional_info[key].append(value)

    # Store the total reward and episode length
    rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_info.append(episode_additional_info)

    if episode % 10 == 0:
        agent.update_target_model()

    tqdm.write(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

#saving the trained model for later
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
filename = f"q_table_{num_episodes}_{formatted_time}.npy"
agent.save(TrainedDQLearning_model, f"DQN_final_{formatted_time}")


# Convert lists to numpy arrays for easier manipulation later
rewards = np.array(rewards)
episode_lengths = np.array(episode_lengths)

#saving the training results for later analysis
filename = f"rewards_{formatted_time}.npy"
if not os.path.exists(TrainedDQLearning_data):
    os.makedirs(TrainedDQLearning_data)
np.save(os.path.join(TrainedDQLearning_data,filename),rewards)

filename = f"episode_length_{formatted_time}.npy"
if not os.path.exists(TrainedDQLearning_data):
    os.makedirs(TrainedDQLearning_data)
np.save(os.path.join(TrainedDQLearning_data,filename), episode_lengths)

with open(os.path.join(TrainedDQLearning_data, f"episode_info_{formatted_time}.pkl"), 'wb') as f:
    pickle.dump(episode_info, f)

#Save training history
np.savez(os.path.join(TrainedDQLearning_data, f"env_history_{formatted_time}.npz"),
         budget_history=env.budget_history,
         sheep_history=env.sheep_history,
         wheat_history=env.wheat_history)

# Save agent training errors
agent.save_training_error(TrainedDQLearning_data, f"training_errors_{formatted_time}")