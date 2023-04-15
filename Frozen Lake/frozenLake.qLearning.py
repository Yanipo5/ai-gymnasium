import math
import collections
import gymnasium as gym
import numpy as np
import statistics
import tqdm
import matplotlib.pyplot as plt

# Solved at episode 15000: average reward: 0.80!
# 15000/50000 [00:10<00:24, 1454.93it/s, mean_reward=0.8]

# Hyperparameters
learning_rate = 1
discount_rate = 0.99
min_episodes_criterion = 50
max_episodes = 80000
epsilon = 1
epsilon_decay = 1 / max_episodes
reward_threshold = 0.8
log_stats_step = 500
demos = 3

# Set seed for experiment reproducibility
is_slippery = False
env_name = "FrozenLake-v1"
map_name = "8x8"
render_mode = None
seed = 42
np.random.seed(seed)

mean_reward = 0

# Store the last min_episodes_criterion episodes rewards & max_x
episodes_reward: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)

# Store statistics of episodes_reward and episodes_max_x
episodes_reward_stats = []

# Create the environment
env = gym.make(env_name, render_mode=render_mode,
               is_slippery=is_slippery, map_name=map_name)


class Qtable():
    def __init__(self, env: gym.Env):
        """Build empty Qtable"""
        self.env = env
        self.qTable = np.zeros([env.observation_space.n, env.action_space.n])

    def __call__(self, state) -> int:
        """Call the model and get action"""
        return np.argmax(self.qTable[state])

    def train(self, state, action, reward, next_state) -> int:
        """Update the current value (Q_v_t: [state_t, action_t]) with the reward, and the expected value (Q_v_t+1) following the policy"""
        optimazed_reward = self.optimaze_reward(reward)
        delta = learning_rate*(optimazed_reward +
                               discount_rate *
                               np.max(self.qTable[next_state])
                               - self.qTable[state, action])
        self.qTable[state, action] += delta

    def updateDone(self, state, action, reward):
        optimazed_reward = self.optimaze_reward(reward)
        self.qTable[state, action] = optimazed_reward

    def plot(self):
        """Plot the Qtable"""
        signs = ['<', 'v', '>', '^']
        colors = [[0.8, 0.8, 0], 'red', 'green', "blue"]  # dark yellow
        n = len(self.qTable)
        s = int(math.sqrt(n))

        for i in range(n):
            action: int = np.argmax(self.qTable[i])
            x = 0 if i == 0 else int(i % s)
            y = int((i - x) / s)
            y = -y

            if (i == n-1):
                marker = '*'
                color = 'orange'
            else:
                marker = signs[action]
                color = colors[action]

            plt.plot(x, y, marker=marker, color=color)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('State/Action Map (Qtable)')
        plt.show()

    def optimaze_reward(self, reward):
        # return 100 if reward == 1 else -1
        return reward


qLearning = Qtable(env)

print(
    f'Training started for: {env_name}, target: {reward_threshold}, (last {min_episodes_criterion} runs).')
max_episodes_tqdm = tqdm.trange(max_episodes)

# Exploring (training episodes)
for episode in max_episodes_tqdm:
    episode_reward = 0
    done = False
    state, info = env.reset()

    # Running steps of episode
    while done is not True:
        # epsilon-greedy policy, explore until epsilon nullifies
        if (epsilon > 0 and np.random.random() < epsilon):
            action = env.action_space.sample()
        else:
            action = qLearning(state)

        # Take action, train, and store update episode values
        next_state, reward, done, truncated, info = env.step(action)
        qLearning.train(state, action, reward, next_state)
        episode_reward += reward

        # Allow for terminal states
        if done:
            qLearning.updateDone(state, action, reward)

        elif truncated:
            done = True

        # Prepare for next step
        state = next_state

    episodes_reward.append(episode_reward)
    epsilon -= epsilon_decay

    if episode % log_stats_step == 0 and episode > 0:
        # Add statistics
        mean_reward = statistics.mean(episodes_reward)
        max_episodes_tqdm.set_postfix(mean_reward=mean_reward)
        episodes_reward_stats.append(mean_reward)

        if mean_reward >= reward_threshold and episode >= min_episodes_criterion:
            break

if (mean_reward >= reward_threshold):
    print(
        f'\nSolved at episode {episode}: average reward: {mean_reward:.2f}!')

    steps = np.array(range(0, len(episodes_reward_stats), 1))
    steps *= log_stats_step
    plt.plot(steps, episodes_reward_stats)
    plt.ylabel('Mean(rewards)')
    plt.xlabel('Episode')
    plt.ylim()
    plt.title('Rewards Progression')
    plt.show()

else:
    print(f'\nNot solved, mean reward: {mean_reward:.2f}!')

env.close()

# Demo (Exploitation of the model)
render_mode = "human"
env = gym.make(env_name, render_mode=render_mode,
               is_slippery=is_slippery, map_name=map_name)
for i in range(demos):
    state, info = env.reset()
    done = False

    while done != True:
        action = qLearning(state)
        state, reward, truncated, done, info = env.step(action)
        if done or truncated:
            done = True

env.close()

# Visualize the Qtable
qLearning.plot()
