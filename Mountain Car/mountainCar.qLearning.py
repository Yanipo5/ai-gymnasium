import collections
import math
import gymnasium as gym
import numpy as np
import statistics
import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
# Small epsilon value for stabilizing division operations
learning_rate = 0.01
discount_rate = 0.99
min_episodes_criterion = 100
max_episodes = 10000
epsillon = 1
epsillon_decay = 1 / max_episodes
discretize_array = [0.1, 0.01]

optimazeReward = True
# Positive multiplier is selected to encourage higher x
optimazeRewardMultiplier = 2

# Set seed for experiment reproducibility
seed = 42
np.random.seed(seed)

# `MountainCar-v0` is considered solved if average reward bigger then -200
reward_threshold = -175
mean_reward = 0

# Create the environment
env_name = "MountainCar-v0"
render_mode = None
env = gym.make(env_name, render_mode=render_mode)


class Qtable():
    def __init__(self, env: gym.Env):
        self.env = env

        temp = (env.observation_space.high -
                env.observation_space.low) / discretize_array
        temp = np.round(temp, 0).astype(int) + 1
        temp = np.append(temp, env.action_space.n)
        self.qTable = np.zeros(temp)

    def __call__(self, state) -> int:
        ds = Qtable.discretizeState(state, self.env)
        return np.argmax(self.qTable[ds[0], ds[1]])

    def train(self, state, action, reward, next_state) -> int:
        """Update the current value (Q_v_t: [state_t, action_t]) with the reward, and the expected value (Q_v_t+1) following the policy"""
        ds1 = Qtable.discretizeState(state, self.env)
        ds2 = Qtable.discretizeState(next_state, self.env)
        optimaze_reward = Qtable.optimazeReward(reward, ds2, self.qTable)
        delta = learning_rate*(optimaze_reward +
                               discount_rate *
                               np.max(self.qTable[ds2[0], ds2[1]])
                               - self.qTable[ds1[0], ds1[1], action])
        self.qTable[ds1[0], ds1[1], action] += delta

    def updateDone(self, state, action, reward):
        ds = Qtable.discretizeState(state, self.env)
        self.qTable[ds[0], ds[1], action] = reward

    @staticmethod
    def discretizeState(state, env: gym.Env):
        """Set the discrete state observetion length"""
        temp = (state - env.observation_space.low) / discretize_array
        return np.round(temp, 0).astype(int)

    @staticmethod
    def optimazeReward(reward: int, discretize_next_state, qTable):
        """Optimaze the rewards for faster learning"""
        if (optimazeReward is False):
            return reward

        return discretize_next_state[0] * optimazeRewardMultiplier - (len(qTable) * optimazeRewardMultiplier + 1)


episodes_reward_stats = []
episodes_max_x_stats = []
# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)
episodes_max_x: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)

qLearning = Qtable(env)

print(
    f'Training started for: {env_name}, target: {reward_threshold}, (last {min_episodes_criterion} runs).')
max_episodes_tqdm = tqdm.trange(max_episodes)

# Running an episode
for episode in max_episodes_tqdm:
    episode_reward = 0
    episode_max_x = env.observation_space.low[0]
    done = False
    state, info = env.reset()

    # Running steps of episode
    while done != True:
        # epsillon-greedy policy, explore until epsillon nullifies
        if (epsillon > 0 and np.random.random() < epsillon):
            action = env.action_space.sample()
            epsillon -= epsillon_decay
        else:
            action = qLearning(state)

        # Take action, and learn
        next_state, reward, done, truncated, info = env.step(action)
        qLearning.train(state, action, reward, next_state)
        episode_reward += reward
        episode_max_x = max(episode_max_x, next_state[0])

        # Prepare for next step
        state = next_state

        # Allow for terminal states
        if done:
            qLearning.updateDone(state, action, 0)

        if done or truncated or next_state[0] >= 0.6:
            done = True

    episodes_reward.append(episode_reward)
    episodes_max_x.append(episode_max_x)

    if episode % 50 == 0 and episode > 0:
        # Add statistics
        mean_reward = statistics.mean(episodes_reward)
        mean_max_x = statistics.mean(episodes_max_x)
        max_episodes_tqdm.set_postfix(
            mean_reward=mean_reward, mean_max_x=mean_max_x)

        episodes_reward_stats.append(mean_reward)
        episodes_max_x_stats.append(mean_max_x)

        if mean_reward > reward_threshold and episode >= min_episodes_criterion:
            break

if (mean_reward > reward_threshold):
    print(
        f'\nSolved at episode {episode}: average reward: {mean_reward:.2f}!')

    steps = range(0, len(episodes_max_x_stats), 1)
    plt.plot(steps, episodes_max_x_stats)
    plt.ylabel('Mean(max_x)')
    plt.xlabel('Episode')
    plt.ylim()
    plt.show()

else:
    print(f'\nNot solved las reward: {mean_reward:.2f}!')

env.close()
# 5900/25000 [01:54<06:10, 51.51it/s, avrange_running_reward=-172, episodes_max_x=0.41]
# Solved at episode 5900: average reward: -172.17!

# Show
render_mode = "human"
env = gym.make(env_name, render_mode=render_mode)
for i in range(10):
    state, info = env.reset()
    done = False

    while done != True:
        action = qLearning(state)
        state, reward, truncated, done, info = env.step(action)

env.close()
