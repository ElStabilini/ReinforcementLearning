import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Define the state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.model = self.build_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.training_error = []
        self.episode_training_error = []
        self.complete_training_error = []

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action(self, state):
        state = np.array(state).reshape(1, -1)
        probs = self.model.predict(state)[0]
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def train(self, states, actions, rewards):
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        with tf.GradientTape() as tape:
            states = np.array(states)
            actions = np.array(actions)
            
            logits = self.model(states)
            action_masks = tf.one_hot(actions, self.action_dim)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
            loss = -tf.reduce_sum(log_probs * discounted_rewards)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        error = loss.numpy()
        self.training_error.append(error)
        self.complete_training_error.append(error)
        self.episode_training_error.append(np.mean(self.training_error))
        self.training_error = []  # Reset for the next episode

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

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

    def save_training_error(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)

        full_path = os.path.join(path, filename + '.npz')

        np.savez(full_path, avg_error=np.array(self.episode_training_error),
                 all_errors=np.array(self.complete_training_error))