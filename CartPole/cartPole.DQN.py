import tensorflow as tf
import tf_agents as tf_agents
import collections
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm
import atexit
import os
import statistics
from skimage.transform import resize
from typing import Any
import gc

# import maths

# Hyperparameters
actions_num = 4
max_episodes = 10_000
epsilon = 1
epsilon_terminal_value = 0.05
epsilon_decay = 1e-5
learning_rate = 1e-3
gamma = 0.99  # Discount factor for past rewards
end_game_reward_shape = -0.5
trainning_epoche_episodes_len = 50
reward_threshold = 50
exploration_frames = 10000
batch_size = 32

# Env Params
env_name = "LunarLander-v2"
render_mode = "rgb_array"
model_file_name = os.path.dirname(__file__) + "/model"
train_model = True
load_weights = True
save_weights = True
seed = 42

# Plot Params
running_reward_interval = 16

# Demo
# max_episodes = 1
# render_mode = "human"
# train_model = False
# trainning_epoche_episodes_len = 10
load_weights = False
save_weights = False
# exploration_frames = 32
# learning_batch_frames = 32
env_name = "CartPole-v1"
# learning_rate = 1e-1
# epsilon = 0
# epsilon_random_episodes = max_episodes * 0.2
# running_reward_interval = 1
# epsilon_decay = (epsilon - epsilon_terminal_value) / (
#     (max_episodes - epsilon_random_episodes) * 0.99
# )
# trainning_epoche_episodes_len = 10
# human_intuition_chance = 0.8


class Observations:
    """Agent observation"""

    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def add(self, state, action, reward, done, next_state) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()


class Model(tf.keras.Sequential):
    """Deep Q-Learning algorithem, updates a trainng model on each batch.
    Train the target model periodically"""

    def __init__(
        self,
        env: gym.Env,
        filters_num=32,
        learning_rate=learning_rate,
    ):
        super().__init__()
        if env.action_space.n is None:  # type: ignore
            raise Exception("missing env.action_space.n")
        if env.observation_space.shape is None:
            raise Exception("missing env.observation_space.shape")

        self.actions = env.action_space.n  # type: ignore

        # Make Sequenctial-visual model accepts n frames sequence
        (states_dims,) = env.observation_space.shape
        input_shape = (None, states_dims)
        self.training_model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=input_shape, dtype=tf.float32, name="input"
                ),
                tf.keras.layers.Dense(filters_num, name="hidden", activation=None),
                tf.keras.layers.Dense(self.actions, name="output", activation=None),
            ]
        )

        self.training_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.Huber(),
        )

        # Training model used for training, and not for taking actions. This would allow a batch update of the model policy.
        self.target_model = tf.keras.models.clone_model(self.training_model)
        self.target_model.summary()

    def get_action(self, state):
        action: tf.Tensor = self._get_action(state)  # type: ignore
        return action.numpy()  # type: ignore

    @tf.function
    def _get_action(self, state) -> tf.Tensor:
        state_tensor = tf.expand_dims(state, 0)
        action_values = self.target_model(state_tensor, training=False)
        action = tf.argmax(action_values, axis=1)
        return action[0]

    # @tf.function
    def train(self, observations: Observations):
        """Train the training model with the observations"""
        # Deep Q-Learning Algorithem
        #   1. Predict future Q values based on next_state
        #   2. Mask actions * observed rewards
        #   3. Calculate updated q values based on the action taken
        #   4. Fit the model based on the taken observation -> [action] = new_q_value
        #   5. Save weights
        # print(f"observations.states: {observations.states}")
        # print("--------")
        # print(f"observations.actions: {observations.actions}")
        # print(f"observations.rewards: {observations.rewards}")
        # print(self.predict(observations.states))
        # test = tf.expand_dims(observations.states[0], 0)
        # print(f"self.target_model(test) {self.training_model(test)}")

        current_states_tensor = tf.constant(observations.states)
        # print(f"current_states_tensor: {current_states_tensor}")

        next_states_tensor = tf.constant(observations.next_states)
        predicated_next_q_values = self.target_model.predict(
            next_states_tensor, verbose="1", batch_size=batch_size
        )

        # predicated_state_q_values = self.target_model.predict(
        #     current_states_tensor, verbose="1", batch_size=batch_size
        # )
        # print(f"predicated_state_q_values: {predicated_state_q_values}")
        # print(f"predicated_next_q_values: {predicated_next_q_values}")

        masks = tf.one_hot(observations.actions, self.actions, dtype=tf.float32)
        rewards = tf.reshape(observations.rewards, shape=(len(observations.rewards), 1))
        reward_mask = tf.multiply(masks, rewards)

        updated_q_values = tf.add(reward_mask, gamma * predicated_next_q_values)
        # print(f"--- pre train---")
        # print(f"current_states_tensor {current_states_tensor}")
        # print(f"updated_q_values {updated_q_values}")
        # print(f"--- pre train---end")
        self.training_model.fit(
            current_states_tensor, updated_q_values, verbose="1", batch_size=batch_size
        )

        # print("self.predict(tf.constant(observations.states))")
        # print(self.predict(tf.constant(observations.states)))
        self.target_model.set_weights(self.training_model.get_weights())

        # print(f"self.target_model(test) {self.training_model(test)}")
        # print(f"updated_q_values {updated_q_values}")

        # predicated_state_q_values = self.target_model.predict(
        #     current_states_tensor, verbose="1", batch_size=batch_size
        # )
        # print(f"predicated_state_q_values: {predicated_state_q_values}")
        # print("--------")

    def load_weights(self, file_name):
        try:
            self.training_model.load_weights(file_name)
            self.target_model.load_weights(file_name)
        except:
            print(f"Loading model wasn't found in: {model_file_name}")

    def save_weights(self, file_name):
        self.target_model.save_weights(file_name)


# Main
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

env = gym.make(env_name, render_mode=render_mode)
env.action_space.seed(seed)
model = Model(env)
if env.spec and env.spec.reward_threshold:
    reward_threshold = env.spec.reward_threshold
print(f"Reward threshold: {reward_threshold}")

# load and save weights
if load_weights:
    model.load_weights(model_file_name)
if save_weights:
    atexit.register(model.save_weights, model_file_name)

episodes_reward: collections.deque[float] = collections.deque(
    maxlen=running_reward_interval
)
rewards: list[float] = []
observation_list = Observations()
total_frames = 0

# Training episode loop
max_episodes_tqdm = tqdm.trange(max_episodes)
for episode in max_episodes_tqdm:
    current_state, info = env.reset()
    done = False
    episode_reward = 0
    step = 0

    # Training frames Loop
    while done != True:
        total_frames += 1
        step += 1

        # epsilon-greedy selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.get_action(current_state)

        # Interact with the environemnt (no action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done is True:
            done = done
            reward = 0
        episode_reward += float(reward)

        # Collect the observations
        observation_list.add(current_state, action, reward, done, next_state)
        current_state = next_state

        # epsilon decay
        if total_frames > exploration_frames and epsilon > epsilon_terminal_value:
            epsilon -= epsilon_decay

        if train_model and total_frames > 0 and total_frames % (batch_size * 100) == 0:
            model.train(observation_list)
            observation_list.clear()
            gc.collect()
            if save_weights:
                model.save_weights(model_file_name)

    # Print post episode
    episodes_reward.append(episode_reward)
    if episode % running_reward_interval == 0:
        rewards.append(statistics.mean(episodes_reward))

    max_episodes_tqdm.set_postfix(
        episode=episode,
        reward=episode_reward,
        epsilon=epsilon,
        total_frames=total_frames,
    )

# Plot Rewards Progression
steps = np.array(range(0, len(rewards), 1))
plt.plot(steps, rewards)
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.ylim()
plt.title("Rewards Progression")
plt.show()
