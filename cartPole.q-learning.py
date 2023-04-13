# Based on: https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df
import numpy as np
import gymnasium as gym
import tqdm


# Hyperparameters
learning_rate = 0.1
discount_factor = 0.95
num_episodes = 100000
epsilon = 1
epsilon_decay_start = 10000
epsilon_decay_value = 0.9995
reward_threshold = 195

total_reward = 0
prior_reward = 0

render_mode = "rgb_array"
# render_mode="human"
env = gym.make("CartPole-v1", render_mode=render_mode)

Observation = [30, 30, 50, 50]
q_table = np.random.uniform(low=0, high=1, size=(
    Observation + [env.action_space.n]))


def get_discrete_state(_state):
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1], dtype=float)
    state = np.array(_state, dtype=float)
    discrete_state = state/np_array_win_size + \
        np.array([15, 10, 1, 10], dtype=float)
    return tuple(discrete_state.astype(int))


t = tqdm.trange(num_episodes)
for episode in t:
    # get the discrete start for the restarted environment
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0  # reward starts as 0 for each episode

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # step action to get new states, reward, and the "done" status.
        new_state, reward, done, trunced, _ = env.step(action)
        episode_reward += reward  # add the reward
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            # update q-table
            current_q = q_table[discrete_state + (action,)]
            max_future_q = np.max(q_table[new_discrete_state])
            new_q = (1 - learning_rate) * current_q + \
                learning_rate * (reward + discount_factor * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05 and episode > epsilon_decay_start and episode_reward > prior_reward:
        epsilon *= epsilon_decay_value

    total_reward += episode_reward  # episode total reward

    if episode % 500 == 0:
        mean_reward = total_reward / 1000
        total_reward = 0

        t.set_postfix(mean_reward=mean_reward, epsilon=epsilon)

    if mean_reward > reward_threshold:
        break

print(f"solved in {episode} episodes (reward_threshold:{reward_threshold})")

env.close()
# view rawCartPole-qLearning hosted with ❤ by GitHub
