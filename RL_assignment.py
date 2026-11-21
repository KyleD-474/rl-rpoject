import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import random
import torch
import torch.nn as nn
import os
import time
import argparse
import csv
import sys

# Set random seeds so training results are reproducible between runs.
SEED = 47

# Seed Python's random number generator.
random.seed(SEED)

# Seed NumPy's random number generator.
np.random.seed(SEED)

# Seed PyTorch's random number generator.
torch.manual_seed(SEED)


# ------------------------------------------------------------------------
# 1) Constants
# ------------------------------------------------------------------------
# The time step for the simulation is set to 1 second, meaning that each step in the dataset represents a 1-second interval.
DELTA_T = 1.0  # seconds

# Start the lead vehicle 20 meters ahead of the ego vehicle for a nice in-between initialization.
INITIAL_DISTANCE = 20.0    # meters

# Follow distance parameters for the reward function.
VERY_MIN_FOLLOW_DISTANCE = 2.0  # meters
MIN_FOLLOW_DISTANCE = 5.0  # meters
MAX_FOLLOW_DISTANCE = 30.0 # meters

# Preferred following distance range within the overall safe zone.
PREFERRED_MIN_FOLLOW_DISTANCE = 15.0  # meters
PREFERRED_MAX_FOLLOW_DISTANCE = 20.0  # meters

# Following distances greater than this are considered excessively far.
WAY_TOO_FAR_FOLLOW_DISTANCE = 45.0  # meters

# Additional penalty applied when the ego vehicle falls excessively far behind.
WAY_TOO_FAR_PENALTY = 25.0

# Penalty applied when the ego vehicle collides with the lead vehicle (i.e., following distance <= 0).
COLLISION_PENALTY = 100.0 

# Weights for the penalties in the reward function.
DISTANCE_PENALTY_WEIGHT = 1.0
SPEED_PENALTY_WEIGHT = 1.0
WAY_TOO_CLOSE_PENALTY_WEIGHT = 10.0
TOO_CLOSE_PENALTY_WEIGHT = 4.0
TOO_FAR_PENALTY_WEIGHT = 2.0
JERK_PENALTY_WEIGHT = 1.0

# The agent is rewarded for maintaining a safe following distance.
SAFE_ZONE_BONUS = 20.0 

# Additional reward for staying in the preferred following distance range.
PREFERRED_DISTANCE_BONUS = 10.0

# The maximum acceleration and deceleration values are set to 2.0 m/s^2 and -2.0 m/s^2, respectively.
MAX_ACCELERATION = 2.0    # m/s^2
MAX_DECELERATION = -2.0   # m/s^2

# Always create a 1200-step speed dataset
DATA_LEN = 1200
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed_profile.csv")

if not os.path.exists(CSV_FILE):
    rng = np.random.default_rng(47)
    speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * rng.standard_normal(DATA_LEN)
    pd.DataFrame({"speed": speeds}).to_csv(CSV_FILE, index=False)
    print(f"Created {CSV_FILE} with {DATA_LEN} steps.")
else:
    print(f"Using existing speed profile: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps."


# Old random speed profile generation code (commented out) for reference. The current implementation uses a pre-generated CSV file for the speed profile.
# CSV_FILE = "speed_profile.csv"
# # Force-generate a 1200-step sinusoidal + noise speed profile
# speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
# df_fake = pd.DataFrame({"speed": speeds})
# df_fake.to_csv(CSV_FILE, index=False)
# print(f"Created {CSV_FILE} with {DATA_LEN} steps.")
# df = pd.read_csv(CSV_FILE)
# full_speed_data = df["speed"].values
# assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# Calculate the speed error, distance error, and reward based on the current state of the ego vehicle and the lead vehicle.
def calculate_errors_and_reward(
    current_speed,
    lead_speed,
    following_distance,
    acceleration,
    previous_acceleration,
    delta_t,
):
    # Calculate the reward based on the difference between current speed and lead speed
    speed_error = abs(current_speed - lead_speed)

    # Unweighted error in meters outside the safe range.
    # 
    # The agent is rewarded for matching the lead vehicle's speed while maintaining a safe following distance. 
    # A target following distance of 5-30 meters is considered acceptable. 
    # If the ego vehicle gets closer than 5 meters or farther than 30 meters, a distance penalty is applied.
    # Also penalize the agent even more for getting too close (less than 2 meters) to the lead vehicle.
    # Too close penalities are weighted more heavily to encourage the agent to maintain a safe following distance.
    if following_distance < MIN_FOLLOW_DISTANCE:
        distance_error = MIN_FOLLOW_DISTANCE - following_distance
    elif following_distance > MAX_FOLLOW_DISTANCE:
        distance_error = following_distance - MAX_FOLLOW_DISTANCE
    else:
        distance_error = 0.0

    # Weighted distance penalty for the reward.
    if following_distance < VERY_MIN_FOLLOW_DISTANCE:
        distance_penalty = WAY_TOO_CLOSE_PENALTY_WEIGHT * (
            VERY_MIN_FOLLOW_DISTANCE - following_distance
        )
    elif following_distance < MIN_FOLLOW_DISTANCE:
        distance_penalty = TOO_CLOSE_PENALTY_WEIGHT * (
            MIN_FOLLOW_DISTANCE - following_distance
        )
    elif following_distance > MAX_FOLLOW_DISTANCE:
        distance_penalty = TOO_FAR_PENALTY_WEIGHT * (
            following_distance - MAX_FOLLOW_DISTANCE
        )
    else:
        distance_penalty = 0.0

    # Apply an additional penalty if the ego vehicle falls excessively far behind.
    # This is to counteract the agent's tendency to "give up" and let the lead vehicle get too far ahead.
    # The collision penalty can also make the agent overly cautious, so this additional penalty encourages the agent to maintain a reasonable following distance.
    if following_distance > WAY_TOO_FAR_FOLLOW_DISTANCE:
        distance_penalty += WAY_TOO_FAR_PENALTY

    distance_penalty *= DISTANCE_PENALTY_WEIGHT

    # The agent is rewarded for maintaining a safe following distance.
    safe_bonus = 0.0
    if MIN_FOLLOW_DISTANCE <= following_distance <= MAX_FOLLOW_DISTANCE:
        safe_bonus = SAFE_ZONE_BONUS
        if PREFERRED_MIN_FOLLOW_DISTANCE <= following_distance <= PREFERRED_MAX_FOLLOW_DISTANCE:
            safe_bonus += PREFERRED_DISTANCE_BONUS

    # The collision penalty is applied if the following distance is less than or equal to zero, 
    # indicating a collision with the lead vehicle.
    collision_penalty = 0.0
    if following_distance <= 0:
        collision_penalty = COLLISION_PENALTY

    # Compute jerk from the previous control action.    
    jerk = abs(acceleration - previous_acceleration) / delta_t
    jerk_penalty = JERK_PENALTY_WEIGHT * jerk

    # Compute the speed penalty based on the absolute speed error
    speed_penalty = SPEED_PENALTY_WEIGHT * speed_error

    reward = (
        safe_bonus
        - speed_penalty
        - distance_penalty
        - collision_penalty
        - jerk_penalty
    )

    return {
        "speed_error": speed_error,
        "distance_error": distance_error,
        "distance_penalty": distance_penalty,
        "jerk": jerk,
        "reward": reward,
    }


# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
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
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Training environment for adaptive cruise control.

    Each episode uses one randomly selected chunk from the lead vehicle's speed
    profile. The observation is [ego speed, lead speed, following distance],
    and the action is the ego vehicle's acceleration, clipped to the configured
    physical limits. The shared `calculate_errors_and_reward` function computes
    the speed and unweighted distance errors, then combines the safe-distance
    bonus with speed, weighted distance, collision, and jerk penalties.
    """

    def __init__(self, episodes_list, delta_t=DELTA_T):
        super().__init__()
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(
            low=MAX_DECELERATION, 
            high=MAX_ACCELERATION, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 100.0], dtype=np.float32),
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
        # ep_idx = np.random.randint(0, self.num_episodes)
        ep_idx = self.np_random.integers(0, self.num_episodes)
        
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Ego vehicle parameters
        self.current_speed = 0.0
        self.current_position = 0.0

        # Acceleration is initialized to zero at the start of each episode.
        self.previous_acceleration = 0.0

        # The lead vehicle's speed is set to the first value in the full dataset
        self.lead_speed = self.current_episode[self.step_idx]

        # The lead vehicle's position is initialized to 20 meters ahead of the ego vehicle.
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

    # This method is called at each timestep of the episode. 
    # It takes an action (acceleration) as input and updates the environment's state accordingly.
    def step(self, action):
        
        # Clamp the selected acceleration to the allowed physical limits before updating the vehicle's speed.
        # I.e. Enforce the acceleration limits defined by the environment.
        acceleration = np.clip(action[0], MAX_DECELERATION, MAX_ACCELERATION)

        # Ego vehicle accelerates according to the agent's action
        self.current_speed += acceleration * self.delta_t

        # Make sure the ego vehicle's speed does not go below zero (i.e., no reverse driving).
        if self.current_speed < 0:
            self.current_speed = 0.0
        
        self.current_position += self.current_speed * self.delta_t

        # Lead vehicle accelerates according to the reference speed profile
        self.lead_speed = self.current_episode[self.step_idx]
        self.lead_position += self.lead_speed * self.delta_t

        # Compute following distance
        self.following_distance = (self.lead_position -self.current_position)

        # Calculate the speed error, distance error, and reward based on the current state of the ego vehicle and the lead vehicle.
        terms = calculate_errors_and_reward(
            current_speed=self.current_speed,
            lead_speed=self.lead_speed,
            following_distance=self.following_distance,
            acceleration=acceleration,
            previous_acceleration=self.previous_acceleration,
            delta_t=self.delta_t,
        )

        # Store the current acceleration for the next timestep.
        self.previous_acceleration = acceleration
        

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
        
        info = {
            "speed_error": terms["speed_error"],
            "distance_error": terms["distance_error"],
            "acceleration": acceleration,
        }
        
        return obs, terms["reward"], terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Evaluation environment for adaptive cruise control.

    Each episode runs through the full lead vehicle speed profile in order.
    The observation is [ego speed, lead speed, following distance], and the
    action is the ego vehicle's acceleration, clipped to the configured
    physical limits. It uses the same `calculate_errors_and_reward` function
    as `TrainEnv`, so evaluation rewards and reported speed and distance errors
    use the same definitions as training.
    """

    def __init__(self, full_data, delta_t=DELTA_T):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(
            low=MAX_DECELERATION, 
            high=MAX_ACCELERATION, 
            shape=(1,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 50.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

        self.idx = 0
        
        # Ego vehicle parameters
        self.current_speed = 0.0
        self.current_position = 0.0

        # Acceleration is initialized to zero at the start of each episode.
        self.previous_acceleration = 0.0

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
        # Clamp the selected acceleration to the allowed physical limits before updating the vehicle's speed.
        # I.e. Enforce the acceleration limits defined by the environment.
        acceleration = np.clip(action[0], MAX_DECELERATION, MAX_ACCELERATION)
        
        # Ego vehicle accelerates according to the agent's action
        self.current_speed += acceleration * self.delta_t

        # Make sure the ego vehicle's speed does not go below zero (i.e., no reverse driving).
        if self.current_speed < 0:
            self.current_speed = 0.0

        self.current_position += self.current_speed * self.delta_t

        # Lead vehicle accelerates according to the reference speed profile
        self.lead_speed = self.full_data[self.idx]
        self.lead_position += self.lead_speed * self.delta_t

        # Compute following distance
        self.following_distance = (self.lead_position - self.current_position)
        
        if self.current_speed < 0:
            self.current_speed = 0.0

        
        # Calculate the speed error, distance error, and reward based on the current state of the ego vehicle and the lead vehicle.
        terms = calculate_errors_and_reward(
            current_speed=self.current_speed,
            lead_speed=self.lead_speed,
            following_distance=self.following_distance,
            acceleration=acceleration,
            previous_acceleration=self.previous_acceleration,
            delta_t=self.delta_t,
        )

        # Store the current acceleration for the next timestep.
        self.previous_acceleration = acceleration
        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([
            self.current_speed, 
            self.lead_speed, 
            self.following_distance
        ], dtype=np.float32)
        
        info = {
            "speed_error": terms['speed_error'],
            "distance_error": terms['distance_error'],
            "acceleration": acceleration,
        }

        return obs, terms['reward'], terminated, truncated, info


# ------------------------------------------------------------------------
# 4) Utility Classes: Logging and Callbacks
# ------------------------------------------------------------------------

# Utility class to log average reward at the end of each episode during training.
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


# Utility class to redirect stdout to both console and a log file.
class ConsoleLogger:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


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
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory where plots and console output will be saved."
    )

    args = parser.parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Redirect stdout to both console and a log file.
    log_file = open(os.path.join(results_dir, "out.txt"), "w")
    sys.stdout = ConsoleLogger(sys.stdout, log_file)

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

    # Seed the training environment.
    train_env.seed(SEED)

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # The policy_kwargs dictionary specifies the architecture of the neural network used for the policy.
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    
    # Original hypoerparmeters for reference:
    # lr = 0.0003
    # g = 0.99
    # t = 0.005

    # Hyperparameters
    lr = 0.0003
    g = 0.99
    t = 0.005

    # Build the SAC model with specified hyperparameters. 
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        batch_size=256,
        buffer_size=200000,
        tau=t,
        gamma=g,
        ent_coef='auto',
        device=device,
        seed=SEED
    )

    # TD3 is deterministic, so add action noise to explore during training.
    # action_noise = NormalActionNoise(
    #     mean=np.zeros(train_env.action_space.shape),
    #     sigma=0.1 * np.ones(train_env.action_space.shape),
    # )

    # model = TD3(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=lr,
    #     buffer_size=200_000,
    #     learning_starts=1_000,
    #     batch_size=256,
    #     tau=t,
    #     gamma=g,
    #     train_freq=1,
    #     gradient_steps=1,
    #     action_noise=action_noise,
    #     device=device,
    #     seed=SEED,
    # )

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
    # save_path = os.path.join(log_dir, f"td3_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    
    # Store testing data for visualization and analysis.
    ego_speeds = []
    lead_speeds = []
    following_distances = []
    speed_errors = []
    distance_errors = []
    accelerations = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ego_speeds.append(obs[0])  # current_speed
        lead_speeds.append(obs[1])  # lead_speed

        signed_speed_error = obs[1] - obs[0]   # lead_speed - ego_speed
        speed_errors.append(signed_speed_error) # signed speed tracking error (positive if ego is slower than lead)

        following_distances.append(obs[2])  # ego-to-lead distance
        
        distance_errors.append(info["distance_error"]) # distance tracking error
        accelerations.append(info["acceleration"]) # acceleration applied by the agent for jerk calculation
        rewards.append(reward)
        if terminated or truncated:
            break

    # Compute jerk statistics.
    # Jerk = rate of change of acceleration.
    # Since delta_t = 1 second, jerk is simply the
    # difference between consecutive acceleration commands.
    accelerations = np.array(accelerations)
    jerks = np.diff(accelerations)  / DELTA_T

    # Prepend a zero so jerk has the same length as the test.
    jerks = np.insert(jerks, 0, 0.0)

    mean_jerk = np.mean(jerks)
    mean_absolute_jerk = np.mean(np.abs(jerks))
    variance_jerk = np.var(jerks)

    # Calculate and display average reward, maximum distance error, mean distance error, and percentage of time within safe following distance.
    avg_test_reward = np.mean(rewards)

    mean_distance_error = np.mean(distance_errors)
    max_distance_error = max(distance_errors)

    mean_speed_error = np.mean(speed_errors) # signed bias
    mean_abs_speed_error = np.mean(np.abs(speed_errors))
    max_abs_speed_error = np.max(np.abs(speed_errors))

    percent_in_safe_zone = 100 * np.mean(np.array(distance_errors) == 0)

    print("--------------------------------------------------------")
    print(f"[TEST] Mean reward per timestep over 1200-step test: {avg_test_reward:.3f}")
    print("--------------------------------------------------------")
    print(f"[TEST] Mean speed error (m/s): {mean_speed_error:.3f}")
    print(f"[TEST] Mean absolute speed error (m/s): {mean_abs_speed_error:.3f}")
    print(f"[TEST] Maximum absolute speed error (m/s): {max_abs_speed_error:.3f}")
    print("--------------------------------------------------------")
    print(f"[TEST] Mean distance error (m): {mean_distance_error:.3f}")
    print(f"[TEST] Maximum distance error (m): {max_distance_error:.3f}")
    print("--------------------------------------------------------")
    print(f"[TEST] Mean jerk (m/s^3): {mean_jerk:.3f}")
    print(f"[TEST] Mean absolute jerk (m/s^3): {mean_absolute_jerk:.3f}")
    print(f"[TEST] Variance of jerk (m/s^3)^2: {variance_jerk:.3f}")
    print("--------------------------------------------------------")
    print(f"[TEST] Vehicle remained within the desired following distance: {percent_in_safe_zone:.2f}% of the test.")
    print("--------------------------------------------------------")

    print("Hyperparameters:")
    print(f"Learning rate: {lr}")
    print(f"Gamma: {g}")
    print(f"Tau: {t}")
    print(f"Chunk size: {chunk_size}")
    print(f"Seed used for RNG: {SEED}")
    # print("Algorithm: TD3")
    print("Algorithm: SAC")


    
    # Plot the lead vehicle and ego vehicle speeds over the entire test.
    #
    # This plot shows how well the ego vehicle tracks the lead vehicle's
    # speed throughout the simulation. Close agreement between the two
    # curves indicates successful speed-following behavior.1.0
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Vehicle Speed Comparison")
    plt.plot(lead_speeds, label="Lead Vehicle Speed", linestyle="--")
    plt.plot(ego_speeds, label="Ego Vehicle Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title("Lead Vehicle Speed vs. Ego Vehicle Speed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "vehicle_speed.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    
    
    # Plot the following distance throughout the simulation.
    #
    # This plot shows the separation distance between the ego vehicle and
    # the lead vehicle. The shaded operating region is defined by the safe
    # following distance bounds in the controller.
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Following Distance")
    plt.plot(following_distances, label="Following Distance")
    plt.axhline(MIN_FOLLOW_DISTANCE, linestyle="--", label="Minimum Safe Distance")
    plt.axhline(MAX_FOLLOW_DISTANCE, linestyle="--", label="Maximum Safe Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title("Following Distance Throughout Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "following_distance.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


    # Plot the speed tracking error throughout the simulation.
    #
    # This plot shows the difference between the ego vehicle speed and
    # the lead vehicle speed. Values near zero indicate good speed tracking.
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Speed Error")
    plt.plot(speed_errors, label="Speed Error")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Perfect Speed Match "
    )
    plt.xlabel("Timestep")
    plt.ylabel("Speed Error (m/s)")
    plt.title("Speed Error Throughout Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "speed_error.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


    # Plot the jerk throughout the simulation.
    #
    # This plot shows the rate of change of acceleration applied by the
    # controller. Smaller jerk values indicate smoother ride comfort.
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("Vehicle Jerk")
    plt.plot(jerks, label="Vehicle Jerk")
    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1,
        label="No Jerk"
    )
    plt.xlabel("Timestep")
    plt.ylabel("Jerk (m/s^3)")
    plt.title("Vehicle Jerk Throughout Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "jerk.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

if __name__ == "__main__":
    main()
