import collections
import gymnasium as gym
import numpy as np
import statistics
import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.1
discount_rate = 0.99
min_episodes_criterion = 100
max_episodes = 5000
epsilon = 1
epsilon_decay = 1 / max_episodes
# `MountainCar-v0` is considered solved if average reward bigger then -200
reward_threshold = -160
discretize_array = [0.1, 0.01]
log_stats_step = 50
demos = 7

# Positive multiplier is selected to encourage higher x
optimazeReward = True

# Set seed for experiment reproducibility
seed = 42
np.random.seed(seed)

mean_reward = 0
mean_max_x = 0

# Store the last min_episodes_criterion episodes rewards & max_x
episodes_reward: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)
episodes_max_x: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)

# Store statistics of episodes_reward and episodes_max_x
episodes_reward_stats = []
episodes_max_x_stats = []

# Create the environment
env_name = "MountainCar-v0"
render_mode = None
env = gym.make(env_name, render_mode=render_mode)


class Qtable():
    def __init__(self, env: gym.Env):
        """Build empty Qtable"""
        self.env = env

        temp_array = (env.observation_space.high -
                      env.observation_space.low) / discretize_array
        temp_array = np.round(temp_array, 0).astype(int) + 1
        temp_array = np.append(temp_array, env.action_space.n)
        self.qTable = np.zeros(temp_array)

    def __call__(self, state) -> int:
        """Call the model and get action"""
        ds = Qtable.discretizeState(state, self.env)
        return np.argmax(self.qTable[ds[0], ds[1]])

    def train(self, state, action, reward, next_state) -> int:
        """Update the current value (Q_v_t: [state_t, action_t]) with the reward, and the expected value (Q_v_t+1) following the policy"""
        ds1 = Qtable.discretizeState(state, self.env)
        ds2 = Qtable.discretizeState(next_state, self.env)
        optimazed_reward = Qtable.optimazeReward(reward, ds2, self.qTable)
        # update the (state_t, action_t) with the optimazed_reward, and the following rewards from following the policy SUM(state_t+1...n, action_t+1...n)
        delta = learning_rate*(optimazed_reward +
                               discount_rate *
                               np.max(self.qTable[ds2[0], ds2[1]])
                               - self.qTable[ds1[0], ds1[1], action])
        self.qTable[ds1[0], ds1[1], action] += delta

    def updateDone(self, state, action, reward):
        ds = Qtable.discretizeState(state, self.env)
        self.qTable[ds[0], ds[1], action] = reward

    def plot(self):
        """Plot the Qtable"""
        signs = ['<', ',', '>']
        colors = ["red", 'grey', "blue"]
        max = self.env.observation_space.high[0]

        for i in range(len(self.qTable)):
            x: float = round(
                i*discretize_array[0] + self.env.observation_space.low[0], 1)
            for j in range(len(self.qTable[0])):
                y: float = round(
                    j*discretize_array[1] + self.env.observation_space.low[1], 2)

                action: int = np.argmax(self.qTable[i][j])
                if (self.qTable[i][j][action] == 0):
                    action = 1

                if (x >= max):
                    marker = '*'
                    color = 'orange'
                else:
                    marker = signs[action]
                    color = colors[action]

                plt.plot(x, y, marker=marker, color=color)

        plt.xlabel('X')
        plt.ylabel('Velocity')
        plt.title('State/Action Map (Qtable)')
        plt.show()

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

        return discretize_next_state[0] - (len(qTable) + reward)


qLearning = Qtable(env)

print(
    f'Training started for: {env_name}, target: {reward_threshold}, (last {min_episodes_criterion} runs).')
max_episodes_tqdm = tqdm.trange(max_episodes)

# Exploring (training episodes)
for episode in max_episodes_tqdm:
    episode_reward = 0
    episode_max_x = env.observation_space.low[0]
    done = False
    state, info = env.reset()

    # Running steps of episode
    while done is not True:
        # epsilon-greedy policy, explore until epsilon nullifies
        if (epsilon > 0 and np.random.random() < epsilon):
            action = env.action_space.sample()
            epsilon -= epsilon_decay
        else:
            action = qLearning(state)

        # Take action, train, and store update episode values
        next_state, reward, done, truncated, info = env.step(action)
        qLearning.train(state, action, reward, next_state)
        episode_reward += reward
        episode_max_x = max(episode_max_x, next_state[0])

        # Prepare for next step
        state = next_state

        # Allow for terminal states
        if done or next_state[0] >= 0.6:
            qLearning.updateDone(state, action, 0)
            done = True

        elif truncated:
            done = True

    episodes_reward.append(episode_reward)
    episodes_max_x.append(episode_max_x)

    if episode % log_stats_step == 0 and episode > 0:
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

    steps = np.array(range(0, len(episodes_max_x_stats), 1))
    steps *= log_stats_step
    plt.plot(steps, episodes_max_x_stats)
    plt.ylabel('Mean(max_x)')
    plt.xlabel('Episode')
    plt.ylim()
    plt.title('Max X Progression')
    plt.show()

else:
    print(f'\nNot solved las reward: {mean_reward:.2f}!')

env.close()
#  1100/10000 [00:19<02:39, 55.67it/s, mean_max_x=0.428, mean_reward=-158]
# Solved at episode 1100: average reward: -158.41!

# Demo (Exploitation of the model)
render_mode = "human"
env = gym.make(env_name, render_mode=render_mode)
for i in range(demos):
    state, info = env.reset()
    done = False

    while done != True:
        action = qLearning(state)
        state, reward, truncated, done, info = env.step(action)
        if done or truncated or state[0] >= 0.6:
            done = True

env.close()

# Visualize the Qtable
qLearning.plot()
