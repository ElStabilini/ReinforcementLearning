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
import datetime
import pickle
import optuna

from pathlib import Path
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers

from environment_v00 import VFAFarmEnv
from algorithms import DQNAgent
from training import train_agent, save_results

#test code must be changed before actually using it

env = gym.make('VFAFarm-v0.0.1')

def create_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 5)
    model = keras.Sequential()
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 32, 256)
        activation = trial.suggest_categorical(f'activation_l{i}', ['relu', 'elu', 'tanh'])
        model.add(layers.Dense(n_units, activation=activation))
    model.add(layers.Dense(env.action_space.n, activation='linear'))
    return model

def objective(trial):
    # Create the model
    model = create_model(trial)
    
    # Suggest values for other hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 256)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.99, 0.9999)
    
    # Create the agent with the suggested hyperparameters
    agent = DQNAgent(env, model=model, learning_rate=learning_rate, 
                     gamma=gamma, epsilon_decay=epsilon_decay, batch_size=batch_size)
    
    # Train the agent
    mean_reward, rewards, episode_lengths, episode_info = train_agent(agent, env, num_episodes=500)
    
    # Save results
    #save_results(agent, rewards, episode_lengths, episode_info)
    
    return mean_reward  # The objective to maximize

# Run the optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))