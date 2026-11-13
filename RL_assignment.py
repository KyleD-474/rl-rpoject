import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
###
# This is creating fake data to simulate a car driving through changing speeds
# Creates 1200 steps of speed data with a sinusoidal pattern and some random noise
# The speed varies between approximately 5 and 15 m/s, with some random fluctuations.
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

INITIAL_DISTANCE = 20.0 # meters

# Force-generate a 1200-step sinusoidal + noise speed profile
speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)

df_fake = pd.DataFrame({"speed": speeds})
df_fake.to_csv(CSV_FILE, index=False)
print(f"Created {CSV_FILE} with {DATA_LEN} steps.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
###
# Instead of using the entire 1200-step dataset as one episode, 
# we can split it into smaller episodes (chunks) of length `chunk_size`. 
#
# This allows the agent to train on shorter sequences, 
# which can help with learning and generalization. 
# The last chunk may be shorter if 1200 is not divisible by `chunk_size`.
#
# Example: chunk_size=100 -> 12 chunks of 100, leftover=0
# 'episodes' contains chunks of length chunk_size, except possibly the last one
# The last chunk may be shorter if 1200 is not divisible by chunk_size
#
# Why chunking matters
# ---------------------
# Short episodes
# Pros
# - learn faster
# - reset more often
# - easier problem
# Cons
# - agent never sees long-term behavior
#
# Long episodes
# Pros
# - more realistic
# Cons
# - harder to learn
# - mistakes accumulate
#
# That's why you're being asked to compare different chunk sizes.
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
#
# This is the heart of the assignment (Reinforcement Learning). 
# The agent will be trained in this environment.
#
# The environment is designed to simulate a speed-following task, 
# where the agent must learn to control acceleration to match a reference speed profile. 
# Each episode consists of a randomly selected chunk of the speed dataset, 
# and the agent receives observations of its current speed and the reference speed, 
# along with rewards based on how closely it follows the reference.
# 
# The environment defines
# - observations
# - actions
# - rewards
# - when an episode ends
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, episodes_list, delta_t=1.0):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([50.0, 50.0, 100.0]),
            dtype=np.float32
        )

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        
        # Ego vehicle parameters
        self.current_speed = 0.0
        self.current_position = 0.0
        
        # Lead vehicle parameters
        self.lead_speed = 0.0
        self.lead_position = 0.0
        
        # Distance between vehicles
        self.following_distance = 0.0

    # This method is called at the start of each episode. 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize ego vehicle parameters
        self.current_speed = 0.0
        self.current_position = 0.0

        # The lead vehicle's speed is set to the first value in the full dataset
        self.lead_speed = self.current_episode[self.step_idx]

        # Start the lead vehicle 20 meters ahead of the ego vehicle.
        # We have a target between 5-30 meters, so we can start at 20 meters
        self.lead_position = INITIAL_DISTANCE

        # Following distance is the distance between the lead vehicle and the ego vehicle
        self.following_distance = (self.lead_position - self.current_position)
        
        # Give the agent its first observation
        obs = np.array([
            self.current_speed, 
            self.lead_speed, 
            self.following_distance
        ], dtype=np.float32)
        info = {}
        return obs, info


    # Think of this method as the "physics" of the environment.
    #
    # It takes the agent's action (acceleration), updates the current speed,
    # calculates the reward based on the difference between the current speed and the reference speed,
    # and returns the new observation, reward, and whether the episode has terminated.
    #
    # It advances the world by one timestep, applying the agent's action to update the current speed.
    #
    #t the step() function is responsible for: 
    # - updating the speed, 
    # - computing the reward, 
    # - advancing to the next time step, and 
    # - telling the agent whether the episode has finished
    def step(self, action):
        # Apply the agent's action (acceleration) to update the current speed
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        
        # Ego vehicle moves
        self.current_position += self.current_speed * self.delta_t

        # Lead vehicle accelerates according to the reference speed profile
        self.lead_speed = self.current_episode[self.step_idx]
        self.lead_position += self.lead_speed * self.delta_t

        # Compute following distance
        self.following_distance = (self.lead_position -self.current_position)
        
        if self.current_speed < 0:
            self.current_speed = 0.0

        # Calculate the reward based on the difference between current speed and reference speed
        error = abs(self.current_speed - self.lead_speed)
        # reward = -error
        reward = -(error ** 2)

        # Increment the step index to move to the next timestep in the episode.
        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False
       
        # Return the new observation, reward, and whether the episode has terminated.
        obs = np.array([
            self.current_speed,
            self.lead_speed,
            self.following_distance
        ], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, full_data, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([50.0, 50.0, 100.0]),
            dtype=np.float32
        )

        self.idx = 0
        
        # Initialize ego vehicle parameters
        self.current_speed = 0.0
        self.current_position = 0.0

        # Lead vehicle parameters
        self.lead_speed = 0.0
        self.lead_position = 0.0

        # Distance between vehicles
        self.following_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        self.current_position = 0.0
        
        # The lead vehicle's speed is set to the first value in the full dataset, and its position is initialized to 20 meters ahead of the ego vehicle. 
        # The following distance is calculated as the difference between the lead vehicle's position and the ego vehicle's position.
        self.lead_speed = self.full_data[self.idx]
        self.lead_position = INITIAL_DISTANCE
        self.following_distance = (self.lead_position - self.current_position)

        obs = np.array([
            self.current_speed,
            self.lead_speed,
            self.following_distance
        ], dtype=np.float32)
        
        info = {}
        return obs, info

    def step(self, action):
        # Ego vehicle accelerates according to the agent's action
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        self.current_position += self.current_speed * self.delta_t

        # Lead vehicle accelerates according to the reference speed profile
        self.lead_speed = self.full_data[self.idx]
        self.lead_position += self.lead_speed * self.delta_t

        # Compute following distance
        self.following_distance = (self.lead_position - self.current_position)
        
        if self.current_speed < 0:
            self.current_speed = 0.0

        # Compute the reward based on the difference between current speed and lead speed
        error = abs(self.current_speed - self.lead_speed)
        # reward = -error
        reward = -(error ** 2)

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([
            self.current_speed, 
            self.lead_speed, 
            self.following_distance
        ], dtype=np.float32)
        
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    args = parser.parse_args()

    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    chunk_size = args.chunk_size
    print(f"[INFO] Using chunk_size = {chunk_size}")

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, delta_t=1.0)

    train_env = DummyVecEnv([make_train_env])

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # The policy_kwargs dictionary specifies the architecture of the neural network used for the policy.
    #
    # In this case, it defines a multi-layer perceptron (MLP) with two hidden layers, 
    # each containing 256 neurons, and using the ReLU activation function.
    #
    # The policy is a neural network that takes the current speed and reference speed 
    # as input and outputs an acceleration action. The architecture of the policy network 
    # is defined by the `policy_kwargs` parameter, which specifies the number of neurons 
    # in each hidden layer and the activation function to be used.
    #
    # The 256,256 is saying that the neural network will have 2 layers with 256 neurons each. 
    # The activation function is ReLU, which is a common choice for hidden layers in neural networks.
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    # Build the SAC model with specified hyperparameters. 
    # The model uses a multi-layer perceptron (MLP) policy with two hidden layers of 256 neurons each 
    # and ReLU activation functions. 
    # The learning rate, batch size, buffer size, tau, gamma, and entropy coefficient are set 
    # to standard values for SAC. 
    # The model is configured to log training progress to the specified logger.
    #
    # A policy is a function that maps observations to actions. 
    # In this case, the policy is a neural network (MLP) that takes the current speed 
    # and reference speed as input and outputs an acceleration action.
    #
    # In other words, a policy is the agent's "brain" that decides what action to take 
    # based on the current state of the environment. (I.e., a rule for choosing actions based on observations.)
    #
    # Multi-Layer Perceptron (MLP) is a type of neural network that consists of multiple layers of neurons,
    # where each layer is fully connected to the next one. (I.e., a standard feedforward neural network.)
    # model = SAC(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=0.0003,
    #     batch_size=256,
    #     buffer_size=200000,
    #     tau=0.005,
    #     gamma=0.99,
    #     ent_coef='auto',
    #     device=device
    # )
    
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=0.0003,
        batch_size=256,
        buffer_size=200000,
        tau=0.005,
        gamma=0.9,
        ent_coef='auto',
        device=device
    )

    model.set_logger(logger)

    total_timesteps = 100_000
    callback = CustomLoggingCallback(log_dir)

    # Train the model using the SAC algorithm for a specified number of timesteps.
    # The training process involves the agent interacting with the environment,
    # collecting experiences, and updating its policy based on the rewards received.
    #
    # The callback is used to log the average reward at the end of each episode,
    # allowing for monitoring of the agent's learning progress over time.
    #
    # The training process is iterative, with the agent continuously improving its policy
    # based on the feedback received from the environment. The goal is to maximize the cumulative reward
    # over the course of training, leading to a policy that effectively follows the reference speed profile.
    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"sac_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    avg_test_reward = np.mean(rewards)
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")

    # Plot the entire test
    # Two lines are plotted: the reference speed (dashed) and the predicted speed (solid).
    #
    # If the two lines overlap closely, it indicates that the agent has learned to 
    # follow the reference speed profile effectively.
    #
    # If the two lines are far apart, it suggests that the agent is not 
    # accurately following the reference speed.
    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on full 1200-step dataset (chunk_size={chunk_size})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
