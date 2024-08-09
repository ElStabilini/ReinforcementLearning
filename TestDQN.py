import gymnasium as gym 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import torch
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
import os
import timeit
import datetime
from pathlib import Path
from tqdm import tqdm

from environment import VFAFarmEnv
from algorithms import DQNAgent

#define path to saving directory
TrainedDQLearning_model = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Model')
TrainedDQLearning_data = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Data')
TrainedDQLearning_plots = Path('/home/elisa/Desktop/Uni/SecondY/RL/SecondPart/SummerClaude/TrainedDQLearning/Plots') 

#initialize agent and environment
env = gym.make('VFAFarm-v0.1')
agent = DQNAgent(env)

agent = DQNAgent(env)
num_episodes = 1000
batch_size = 32

# Lists to store rewards and episode lengths
rewards = []
episode_lengths = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    # Store the total reward and episode length
    rewards.append(total_reward)
    episode_lengths.append(steps)

    if episode % 10 == 0:
        agent.update_target_model()

    tqdm.write(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")
    print(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

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
