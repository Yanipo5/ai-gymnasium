import gymnasium as gym
import numpy as np
import tqdm

# Hyperparameters
learning_rate = 0.1
discount_rate = 0.99
min_episodes_criterion = 100
max_episodes = 5000
epsillon = 1
epsillon_decay = 1 / max_episodes
# `MountainCar-v0` is considered solved if average reward bigger then -200
reward_threshold = -170
discretize_array = [0.1, 0.01]

# Positive multiplier is selected to encourage higher x
optimazeReward = True
optimazeRewardMultiplier = 2

# Set seed for experiment reproducibility
seed = 42
np.random.seed(seed)

# Create the environment
env_name = "MountainCar-v0"
render_mode = None
env = gym.make(env_name, render_mode=render_mode)


class Qtable():
    def __init__(self, env: gym.Env):
        self.env = env

        temp_array = (env.observation_space.high -
                      env.observation_space.low) / discretize_array
        temp_array = np.round(temp_array, 0).astype(int) + 1
        temp_array = np.append(temp_array, env.action_space.n)
        self.qTable = np.zeros(temp_array)

    def __call__(self, state) -> int:
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


qLearning = Qtable(env)

print(
    f'Training started for: {env_name}, target: {reward_threshold}, (last {min_episodes_criterion} runs).')
max_episodes_tqdm = tqdm.trange(max_episodes)

# Running an episode
for episode in max_episodes_tqdm:
    episode_reward = 0
    done = False
    state, info = env.reset()

    # Running steps of episode
    while done is not True:
        # epsillon-greedy policy, explore until epsillon nullifies
        if (epsillon > 0 and np.random.random() < epsillon):
            action = env.action_space.sample()
            epsillon -= epsillon_decay
        else:
            action = qLearning(state)

        # Take action, train, and store update episode values
        next_state, reward, done, truncated, info = env.step(action)
        qLearning.train(state, action, reward, next_state)
        episode_reward += reward

        # Prepare for next step
        state = next_state

        # Allow for terminal states
        if done or next_state[0] >= 0.6:
            qLearning.updateDone(state, action, 0)
            done = True

        elif truncated:
            done = True

    if episode_reward > reward_threshold and episode >= min_episodes_criterion:
        break

print(
    f'\nSolved at episode {episode}: episode reward: {episode_reward:.2f}!')
# Solved at episode 392: episode reward: -163.00!

env.close()
