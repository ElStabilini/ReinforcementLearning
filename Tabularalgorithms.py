import gymnasium as gym 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 
import torch
import matplotlib.pyplot as plt
import random
import os

from IPython import display
from tqdm import tqdm
from collections import defaultdict
from collections import deque
from random import sample
from sklearn.preprocessing import PolynomialFeatures

class TabularQLearningAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((env.observation_space.nvec[0],
                                 env.observation_space.nvec[1],
                                 env.observation_space.nvec[2],
                                 env.action_space.n))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[tuple(state)])

    def train(self, state, action, reward, next_state, done):
        current_q = self.q_table[tuple(state) + (action,)]
        if not done:
            max_next_q = np.max(self.q_table[tuple(next_state)])
            new_q = reward + self.gamma * max_next_q
        else:
            new_q = reward
        
        self.q_table[tuple(state) + (action,)] += self.learning_rate * (new_q - current_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path,filename), self.q_table)

    def load(self, path, filename):
        self.q_table = np.load(os.path.join(path,filename))


'''
        NOTES ON THE QLEARNING AGENT

Now, let's discuss the differences and improvements:

1. Q-table vs Neural Network: The main difference is that we're using a Q-table instead of a neural network. This is more appropriate for discrete, tabular environments like the one we created.
2. State representation: We're using the discrete state directly, which maps to indices in our Q-table. This is more efficient for tabular methods.
3. Action selection: The act method is similar, but it now uses the Q-table instead of a neural network prediction.
4. Training: The train method now updates the Q-table directly using the Q-learning update rule, rather than fitting a neural network.
5. Simplicity: This implementation is much simpler and more suited to the tabular environment we created.

Regarding your implementation:

1. It's not incorrect, but it's overkill for a simple tabular environment. Your approach would be more suitable for complex environments with continuous state spaces.
2. Using a neural network for a simple problem can lead to slower learning and potential overfitting.
3. The _build_model method in your implementation creates a neural network, which isn't necessary for tabular Q-learning.
4. Your train method uses batch training with the neural network, which is more complex than needed for this problem.        
'''

class TabularSARSALambdaAgent:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, lambda_=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lambda_ = lambda_

        # Initialize Q-table
        self.q_table = np.zeros((env.observation_space.nvec[0],
                                 env.observation_space.nvec[1],
                                 env.observation_space.nvec[2],
                                 env.action_space.n))
        
        # Initialize eligibility traces
        self.e_traces = np.zeros_like(self.q_table)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[tuple(state)])

    def train(self, state, action, reward, next_state, next_action, done):
        state_action = tuple(state) + (action,)
        next_state_action = tuple(next_state) + (next_action,)

        # Calculate TD error
        td_target = reward + self.gamma * self.q_table[next_state_action] * (1 - done)
        td_error = td_target - self.q_table[state_action]

        # Update eligibility traces
        self.e_traces *= self.gamma * self.lambda_
        self.e_traces[state_action] += 1

        # Update Q-values
        self.q_table += self.learning_rate * td_error * self.e_traces

        if done:
            self.e_traces *= 0  # Reset traces at end of episode

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)

'''
    NOTES ON SARSA(λ) agent

    1. Eligibility trace matrix was added: e_traces that's the same shape as the Q-table.
    2. The train method now updates all state-action pairs according to their eligibility.
    3. The λ parameter was added to control the decay of eligibility traces.
    4. The training loop now selects the next action before calling train.

Using TD(λ) in this tabular setting could potentially lead to faster learning and better performance, especially if there are delayed rewards in your farm management scenario. 
'''

class VFAQLearningAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.n_actions = env.action_space.n
        self.n_features = 10  # Number of features after polynomial expansion

        # Initialize feature transformer
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        
        # Initialize weights
        self.weights = np.zeros((self.n_actions, self.n_features))

    def get_features(self, state):
        # Normalize state
        normalized_state = state / self.env.observation_space.high
        # Transform state to polynomial features
        return self.poly.fit_transform(normalized_state.reshape(1, -1)).flatten()

    def get_q_value(self, state, action):
        features = self.get_features(state)
        return np.dot(self.weights[action], features)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        features = self.get_features(state)
        current_q = self.get_q_value(state, action)
        
        if not done:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.n_actions)]
            max_next_q = np.max(next_q_values)
            target = reward + self.gamma * max_next_q
        else:
            target = reward

        # Update weights
        self.weights[action] += self.learning_rate * (target - current_q) * features

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, filename), self.weights)

    def load(self, path, filename):
        self.weights = np.load(os.path.join(path, filename))

    def print_feature_names(self):
        # Use a dummy state to fit the PolynomialFeatures
        dummy_state = np.zeros(self.env.observation_space.shape)
        self.get_features(dummy_state)  # This fits the PolynomialFeatures

        # Get feature names
        feature_names = self.poly.get_feature_names(['budget', 'sheep', 'year'])
        
        print("Feature names:")
        for i, name in enumerate(feature_names):
            print(f"{i}: {name}")

    def print_feature_values(self, state):
        features = self.get_features(state)
        feature_names = self.poly.get_feature_names(['budget', 'sheep', 'year'])
        
        print("Feature values:")
        for name, value in zip(feature_names, features):
            print(f"{name}: {value}")


#The value function approximation agent is in this folder only because for clarity I needed to keep only the DQNAgent classe in algorithms.py
# VFAgent isn't actually useful because it's an agent defined in condition of deadly triad (check)