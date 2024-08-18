import numpy as np
from tqdm import tqdm
import datetime
import os
import csv
import pickle
from pathlib import Path

def train_agent(agent, env, num_episodes=1000, batch_size=32, update_target_every=10):
    rewards = []
    episode_lengths = []
    episode_info = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_additional_info = {}

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            #agent.remember(state, action, reward, next_state, done)
            #agent.replay()
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1

            # Accumulate info for the episode
            for key, value in info.items():
                if key not in episode_additional_info:
                    episode_additional_info[key] = []
                episode_additional_info[key].append(value)

        rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_info.append(episode_additional_info)

        if episode % update_target_every == 0:
            agent.update_target_model()

        if episode % 100 == 0:  # Print less frequently to reduce output
            tqdm.write(f"Episode: {episode}, Total Reward: {total_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")

    return np.mean(rewards), np.array(rewards), np.array(episode_lengths), episode_info



def save_results(agent, path, rewards, episode_lengths, episode_info):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    
    # Define paths
    TrainedDQLearning_model = path/'Model'
    TrainedDQLearning_data = path/'Data'

    TrainedDQLearning_model.mkdir(parents=True, exist_ok=True)
    TrainedDQLearning_data.mkdir(parents=True, exist_ok=True)
    
    # Save model
    agent.save(TrainedDQLearning_model, f"DQN_final_{formatted_time}")
    
    # Save rewards and episode lengths
    np.save(os.path.join(TrainedDQLearning_data, f"rewards_{formatted_time}.npy"), rewards)
    np.save(os.path.join(TrainedDQLearning_data, f"episode_length_{formatted_time}.npy"), episode_lengths)
    
    # Save episode info
    with open(os.path.join(TrainedDQLearning_data, f"episode_info_{formatted_time}.pkl"), 'wb') as f:
        pickle.dump(episode_info, f)

    # Save agent training errors
    agent.save_training_error(TrainedDQLearning_data, f"training_errors_{formatted_time}")
    
    return formatted_time




def save_hyperparameters(path, hyperparameters):
    file_path = path / 'hyperparameters.csv'
    file_exists = file_path.exists()

    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['learning_rate', 'gamma', 'epsilon_decay', 'batch_size', 'filename'])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(hyperparameters)