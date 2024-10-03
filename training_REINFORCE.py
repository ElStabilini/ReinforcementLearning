from pathlib import Path
import gymnasium as gym
import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm

from environment_v01 import VFAFarmEnv
from REINFORCE_agent import REINFORCEAgent  # Assume this is where you've defined the REINFORCEAgent

# Define paths to saving directories
TrainedREINFORCE_model = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedREINFORCE/Model')
TrainedREINFORCE_data = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedREINFORCE/Data')
TrainedREINFORCE_plots = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedREINFORCE/Plots')

# Initialize agent and environment
env = gym.make('VFAFarm-v0.0.1')
agent = REINFORCEAgent(env)

num_episodes = 1000

# Lists to store rewards, episode lengths, and other data
rewards = []
episode_lengths = []
episode_info = []

for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    episode_additional_info = {}

    # Lists to store episode data
    states = []
    actions = []
    rewards_episode = []

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store step data
        states.append(state)
        actions.append(action)
        rewards_episode.append(reward)

        state = next_state
        total_reward += reward
        steps += 1

        # Accumulate info for the episode
        for key, value in info.items():
            if key not in episode_additional_info:
                episode_additional_info[key] = []
            episode_additional_info[key].append(value)

    # Train the agent after the episode is complete
    agent.train(states, actions, rewards_episode)
    agent.complete_training_error.extend(agent.training_error)

    # Store the total reward and episode length
    rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_info.append(episode_additional_info)

    tqdm.write(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}")

# Saving the trained model for later
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
agent.save(TrainedREINFORCE_model, f"REINFORCE_final_{formatted_time}")

# Convert lists to numpy arrays for easier manipulation later
rewards = np.array(rewards)
episode_lengths = np.array(episode_lengths)

# Saving the training results for later analysis
if not os.path.exists(TrainedREINFORCE_data):
    os.makedirs(TrainedREINFORCE_data)

np.save(os.path.join(TrainedREINFORCE_data, f"rewards_{formatted_time}.npy"), rewards)
np.save(os.path.join(TrainedREINFORCE_data, f"episode_length_{formatted_time}.npy"), episode_lengths)

with open(os.path.join(TrainedREINFORCE_data, f"episode_info_{formatted_time}.pkl"), 'wb') as f:
    pickle.dump(episode_info, f)

# Save training history
np.savez(os.path.join(TrainedREINFORCE_data, f"env_history_{formatted_time}.npz"),
         budget_history=env.unwrapped.budget_history,
         sheep_history=env.unwrapped.sheep_history,
         wheat_history=env.unwrapped.wheat_history)

# Save agent training errors
agent.save_training_error(TrainedREINFORCE_data, f"training_errors_{formatted_time}")