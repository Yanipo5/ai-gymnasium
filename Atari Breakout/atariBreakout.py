import tensorflow as tf
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

# Hyperparameters
filters = 32
actions_num = 4
kernel_size = (4, 4)
strides = (2, 2)
frames_to_skip = 4
frames_memory_length = 4
max_episodes = 200
epsilon = 1
epsilon_decay = 1 / (max_episodes * 0.8)
epsilon_terminal_value = 0.01
learning_rate = 1e-3
gamma = 0.99  # Discount factor for past rewards
end_game_reward_shape = -0.5
train_episodes_interval = 20

# Env Params
env_name = "ALE/Breakout-v5"
render_mode = "rgb_array"
repeat_action_probability = 0
obs_type = "grayscale"
model_file_name = os.path.dirname(__file__) + "/model"
train_model = True
load_weights = True
save_weights = True

# Plot Params
running_reward_interval = 16

# Demo
max_episodes = 1
epsilon = 0
render_mode = "human"
# train_model = False
# load_weights = False
# save_weights = False


class FramesState(collections.deque):
    """Stores a series of frames"""

    reward: float = 0

    def __init__(self, maxlen=frames_memory_length):
        super().__init__(maxlen=maxlen)

    def add_frame(self, frame: np.ndarray, reward):
        """Reduce the image center to 80x80 frame"""
        cropped_frame = frame[33:-17]
        img_resized = resize(cropped_frame, output_shape=(80, 80), anti_aliasing=True)
        super().append(img_resized)
        self.reward += reward

    def reset(self):
        self.clear()
        super().append(np.zeros((80, 80)))
        super().append(np.zeros((80, 80)))
        super().append(np.zeros((80, 80)))
        super().append(np.zeros((80, 80)))
        self.reward = 0

    def copyFromFrame(self, framesState):
        # TODO fix rewards update
        super().append(framesState[0])
        super().append(framesState[1])
        super().append(framesState[2])
        super().append(framesState[3])
        self.reward = framesState.reward

    def copy(self):
        return [self[0].copy(), self[1].copy(), self[2].copy(), self[3].copy()]

    def clear(self):
        self.reward = 0
        super().clear()

    def getTensor(self) -> tf.Tensor:
        return tf.expand_dims(self.copy(), axis=0)


class FrameHistory:
    """This calss stores the history of frames.
    Used for Batch update of the model"""

    current_states: list[FramesState] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_states: list[FramesState] = []

    def __init__(self):
        pass

    def collect(self, current_state, action, reward, next_state):
        self.current_states.append(current_state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state.copy())

    def clear(self):
        self.current_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()


class Model(tf.keras.Sequential):
    """Deep Q-Learning algorithem, updates a trainng model on each batch.
    Train the target model periodically"""

    def __init__(self):
        super().__init__()

        # Make Sequenctial-visual model accepts n frames sequence
        self.training_model = tf.keras.Sequential(
            [
                tf.keras.layers.ConvLSTM2D(
                    filters,
                    kernel_size=8,
                    input_shape=(4, 80, 80, 1),
                    padding="same",
                    strides=4,
                ),
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=4,
                    strides=2,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.Conv2D(
                    filters,
                    kernel_size=3,
                    strides=1,
                    activation=tf.keras.activations.relu,
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(filters, activation=tf.keras.activations.relu),
                tf.keras.layers.Dense(
                    actions_num,
                    activation=tf.keras.activations.linear,
                    name="output-layer",
                ),
            ]
        )
        self.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.Huber(),
        )
        # Training model used for training, and not for taking actions. This would allow a batch update of the model policy.
        self.target_model = tf.keras.models.clone_model(self.training_model)

        self.training_model.summary()

    def get_action(self, frames_state: FramesState):
        time_series = frames_state.getTensor()
        action = self._get_action(time_series)
        return action.numpy()  # type: ignore

    @tf.function
    def _get_action(self, frames_state_tensor):
        # Inner @tf.function for increased performence
        action_probs = self.training_model(frames_state_tensor, training=False)
        action = tf.argmax(action_probs, axis=1)
        return action[0]

    def train(self, reply_history: FrameHistory):
        """Train the training model with the observations"""
        # Deep Q-Learning Algorithem
        #   1. Predict future Q values based on next_state
        #   2. Mask actions * observed rewards
        #   3. Calculate updated q values based on the action taken
        #   4. Fit the model based on the taken observation -> [action] = new_q_value

        next_states_tensor = Model.get_state_tensor(reply_history.next_states)
        predicated_next_q_values = self.target_model.predict(
            next_states_tensor, verbose="1"
        )

        masks = tf.one_hot(
            reply_history.actions, actions_num, dtype=tf.float32, name="masks"
        )
        rewards = tf.constant(reply_history.rewards, dtype=tf.float32, name="rewards")
        rewards = tf.reshape(rewards, shape=(len(rewards), 1))
        reward_mask = tf.multiply(masks, rewards, name="reward_mask")

        updated_q_values = tf.add(reward_mask, gamma * predicated_next_q_values)

        current_states_tensor = tf.convert_to_tensor(
            reply_history.current_states, dtype=tf.float32, name="current_states_tensor"
        )

        self.training_model.fit(current_states_tensor, updated_q_values, verbose="1")

    def update_weights(self):
        self.target_model.set_weights(self.training_model.get_weights())

    def load_weights(self, file_name):
        try:
            self.training_model.load_weights(file_name)
            self.target_model.load_weights(file_name)
        except:
            print(f"Loading model wasn't found in: {model_file_name}")

    def save_weights(self, file_name):
        self.target_model.save_weights(file_name)

    @staticmethod
    def get_state_tensor(state):
        return map(lambda ns: tf.expand_dims(ns, axis=0), state)


# Main
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

model = Model()

# load and save weights
if load_weights:
    model.load_weights(model_file_name)
if save_weights:
    atexit.register(model.save_weights, model_file_name)

current_frames = FramesState(maxlen=frames_memory_length)
prev_frames = FramesState(maxlen=frames_memory_length)

frames_history = FrameHistory()

env = gym.make(
    env_name,
    render_mode=render_mode,
    repeat_action_probability=repeat_action_probability,
    obs_type=obs_type,
)

total_frames = 0
episodes_reward: collections.deque[int] = collections.deque(
    maxlen=running_reward_interval,
)
rewards: list[float] = []

# Training episode loop
max_episodes_tqdm = tqdm.trange(max_episodes)
for episode in max_episodes_tqdm:
    current_frames.clear()
    prev_frames.clear()
    frame, info = env.reset()

    current_frames.add_frame(frame, 0)
    action = 1  # fire the ball
    done = False
    episode_reward = 0
    step = 0

    # Training frames Loop
    while done != True:
        total_frames += 1
        step += 1
        if step % frames_to_skip != 0:
            # Interact with the environemnt (no action)
            frame, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated
            episode_reward += reward

            if done and end_game_reward_shape is not False:
                # Reward shaping, if the game is done, reduce the reward (less then 1, incase the game over due to braking the last break)
                reward += end_game_reward_shape

            current_frames.add_frame(frame, reward)

            if done is False:
                continue
            else:
                # Collect the end game observation (no additional actions, and rewards)
                while len(current_frames) < frames_memory_length:
                    current_frames.add_frame(frame, 0)
                frames_history.collect(
                    prev_frames, action, current_frames.reward, current_frames
                )
            continue

        # Collect the experience into batch, and prepare for next step
        if len(prev_frames) > 0:
            frames_history.collect(
                prev_frames, action, prev_frames.reward, current_frames
            )
        prev_frames.clear()
        prev_frames.copyFromFrame(current_frames)

        # epsilon-greedy selection
        if epsilon > 0 and np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.get_action(current_frames)

        # Take action and prepare next series
        current_frames.clear()
        frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        current_frames.add_frame(frame, reward)
        episode_reward += reward

    # Print post episode
    episodes_reward.append(int(episode_reward))
    if episode % running_reward_interval == 0:
        rewards.append(statistics.mean(episodes_reward))

    # epsilon decay
    if epsilon > 0:
        epsilon -= epsilon_decay
        if epsilon < epsilon_terminal_value:
            epsilon = 0

    if train_model and episode > 0 and episode % train_episodes_interval == 0:
        model.train(frames_history)
        model.update_weights()
        frames_history.clear()
        model.save_weights(model_file_name)

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
