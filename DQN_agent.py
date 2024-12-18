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

from IPython import display
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from collections import defaultdict
from collections import deque
from random import sample
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures

from tensorflow import keras
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, env, model=None, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, memory_size=10000):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.step_errors = []  # Store errors for each training step
        self.episode_errors = []  # Store average error for each episode
        self.current_episode_errors = []  # Temporary storage for current episode
        self.episode_done = False  # Flag to track episode completion

        # Define the state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Initialize Q-network
        if model is None:
            self.model = self.build_model()
        else:
            self.model = model

        # Initialize target network
        self.target_model = keras.models.clone_model(self.model)
        self.update_target_model()

    def build_model(self):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        state = np.array(state).reshape(1, -1)  # Ensure state is a 2D numpy array
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state, done):
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)

        target = self.model.predict(state)[0]
        old_val = target[action]

        if done:
            target[action] = reward
        else:
            t = self.target_model.predict(next_state)[0]
            target[action] = reward + self.gamma * np.amax(t)

        # Calculate TD error
        td_error = target[action] - old_val
        self.step_errors.append(td_error)
        self.current_episode_errors.append(td_error)

        self.model.fit(state, np.array([target]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if done:
            self.finalize_episode()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        batch_errors = []
        contains_episode_end = False

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state[np.newaxis, ...])[0]
            old_val = target[action]
            
            if done:
                target[action] = reward
                contains_episode_end = True
            else:
                t = self.target_model.predict(next_state[np.newaxis, ...])[0]
                target[action] = reward + self.gamma * np.amax(t)

            # Calculate TD error
            td_error = target[action] - old_val
            batch_errors.append(td_error)

            states.append(state)
            targets.append(target)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Store the average error for this batch
        avg_batch_error = np.mean(batch_errors)
        self.step_errors.append(avg_batch_error)
        self.current_episode_errors.append(avg_batch_error)

        # If the batch contains an episode end, finalize the episode
        if contains_episode_end and not self.episode_done:
            self.finalize_episode()
            self.episode_done = True
        elif not contains_episode_end:
            self.episode_done = False

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def finalize_episode(self):
        """Helper method to handle episode completion"""
        if len(self.current_episode_errors) > 0:
            episode_avg_error = np.mean(self.current_episode_errors)
            self.episode_errors.append(episode_avg_error)
            self.current_episode_errors = []

    def train_episode(self, max_steps=1000):
        state = self.env.reset()[0]
        total_reward = 0
        done = False
        truncated = False
        step = 0
        self.episode_done = False  # Reset episode done flag
        
        while not (done or truncated) and step < max_steps:
            action = self.act(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Store experience in memory
            self.remember(state, action, reward, next_state, done)
            
            # Train without replay
            self.train(state, action, reward, next_state, done)
            
            # Perform experience replay
            self.replay()
            
            state = next_state
            total_reward += reward
            step += 1
        
        return total_reward

    def save(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, filename + '.keras')  # Add .keras extension
        try:
            self.model.save(full_path)
            print(f"Model saved successfully to {full_path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            # Attempt to save in TensorFlow SavedModel format
            try:
                tf.saved_model.save(self.model, full_path)
                print(f"Model saved in SavedModel format to {full_path}")
            except Exception as e:
                print(f"Error saving model in SavedModel format: {str(e)}")

    def load(self, path, filename):
        full_path = os.path.join(path, filename + '.keras')  # Add .keras extension
        if not os.path.exists(full_path):
            full_path = os.path.join(path, filename)  
            if not os.path.exists(full_path):
                print(f"No model file found at {full_path}")
                return
    
        try:
            self.model = keras.models.load_model(full_path)
            print(f"Model loaded successfully from {full_path}")
        except Exception as e:
            print(f"Error loading Keras model: {str(e)}")
            try:
                self.model = tf.saved_model.load(full_path)
                print(f"Model loaded from SavedModel format at {full_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return
    
        self.update_target_model()    

    def save_training_error(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        
        full_path = os.path.join(path, filename + '.npz')
        
        np.savez(full_path, 
                 step_errors=np.array(self.step_errors),
                 episode_errors=np.array(self.episode_errors))