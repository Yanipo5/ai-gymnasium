import tensorflow as tf
import collections
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm
from skimage.transform import resize
import atexit
from typing import Any


def do_something_on_exit():
    # code to run before exit
    print("Exiting program...")


atexit.register(do_something_on_exit)


# Hyperparameters
filters = 32
actions = 4
kernel_size = (4, 4)
strides = (2, 2)
frames_to_skip = 5
frames_memory_length = 2
max_episodes = 25
episodes_learning_batch = 5
epsilon = 1
epsilon_decay = 1 / (max_episodes * 0.8)
learning_rate = 1e-3
gamma = 0.99  # Discount factor for past rewards


# Hyperparameters (env)
repeat_action_probability = 0
render_mode = "rgb_array"
# render_mode = "human"
obs_type = "grayscale"
# -------------------------------------------------

# Make env
env = gym.make(
    "ALE/Breakout-v5",
    render_mode=render_mode,
    repeat_action_probability=repeat_action_probability,
    obs_type=obs_type,
)


class Model(tf.keras.Sequential):
    def __init__(self):
        super().__init__()

        # Make Sequenctial-visual model accepts n frames sequence
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.ConvLSTM2D(
                    filters,
                    kernel_size=4,
                    input_shape=(2, 80, 80, 1),
                    padding="same",
                    strides=4,
                ),
                tf.keras.layers.Conv2D(filters, kernel_size=2, strides=2),
                tf.keras.layers.Conv2D(filters * 2, kernel_size=2, strides=1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    filters * 2, activation=tf.keras.activations.relu
                ),
                tf.keras.layers.Dense(
                    actions,
                    activation=tf.keras.activations.softmax,
                    name="output-layer",
                ),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=1.0
            ),
            loss=tf.keras.losses.Huber(),
        )
        # Training model used for training, and not for taking actions. This would allow a batch update of the model policy.
        self.clone = tf.keras.models.clone_model(self.model)

        self.model.summary()

    def __call__(self, inputs, training=None, mask=None):
        return self.model(inputs)
    
    def train(self, reward, time_series: tf.Tensor):
        # Calculate expected Q value based on the stable model
        stable_action_probs = self.clone(time_series)
        stable_q_action = tf.reduce_max(stable_action_probs, axis=1)
        expected_stable_reward = reward + gamma * stable_q_action

        # Calculate loss gradients
        with tf.GradientTape() as tape:
            # Train the model with the state
            action_probs = self.model(time_series)
            q_action = tf.reduce_max(action_probs, axis=1)
            loss = self.model.loss(expected_stable_reward, q_action)  # type: ignore

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_weights(self):
        self.model.set_weights(self.clone.get_weights())


class FramesState(collections.deque):
    def __init__(self, maxlen=frames_memory_length):
        super().__init__(maxlen=maxlen)

    def addFrame(self, observation):
        """Reduce the image center to 80x80 frame"""
        cropped_observation = observation[33:-17]
        img_resized = resize(
            cropped_observation, output_shape=(80, 80), anti_aliasing=True
        )
        self.append(img_resized)

    def reset(self):
        self.clear()
        self.append(np.zeros((80, 80)))
        self.append(np.zeros((80, 80)))

    def getTensor(self) -> tf.Tensor:
        return tf.expand_dims([self[0], self[1]], axis=0)


# Main
model = Model()
frames_state = FramesState(maxlen=frames_memory_length)
total_frames = 0
rewards = []
max_episodes_tqdm = tqdm.trange(max_episodes)

# Training Loop (episode, frame)
for episode in max_episodes_tqdm:
    env.reset()
    frames_state.reset()
    done = False
    episode_reward = 0
    step = 0

    while done != True:
        total_frames += 1
        step += 1
        if step % frames_to_skip != 0:
            env.step(0)
            continue

        # epsilon-greedy selection
        if epsilon > 0 and np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            time_series = frames_state.getTensor()
            action_probs: Any = model(time_series)
            action = tf.argmax(action_probs[0]).numpy()

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        max_episodes_tqdm.set_postfix(
            episode=episode,
            step=step,
            episode_reward=episode_reward,
            epsilon=epsilon,
            total_frames=total_frames,
        )

        model.train(reward=reward, time_series=frames_state.getTensor())
        
        # Prepare next step state
        frames_state.addFrame(observation)

        if episode > 1 and episode % 1000 == 0:
            model.update_weights()

    rewards.append(episode_reward)

    # decay epsilon on
    epsilon -= epsilon_decay
    if epsilon < 0.01:
        epsilon = 0


steps = np.array(range(0, len(rewards), 1))
plt.plot(steps, rewards)
plt.ylabel("Reward")
plt.xlabel("Episode")
plt.ylim()
plt.title("Rewards Progression")
plt.show()
