import tensorflow as tf
import collections
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import tqdm
from skimage.transform import resize

# Hyperparameters
filters = 32
actions = 4
kernel_size = (4, 4)
strides = (2, 2)
frames_to_skip = 15
frames_memory_length = 2
max_episodes = 10
episodes_learning_batch = 5
epsilon = 1
epsilon_decay = 1 / max_episodes
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

# Make Sequenctial-visual model accepts n frames sequence
model = tf.keras.Sequential(
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
        tf.keras.layers.Dense(filters * 2, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(
            actions, activation=tf.keras.activations.softmax, name="output-layer"
        ),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
    loss=tf.keras.losses.Huber(),
)
model.summary()
# Training model used for training, and not for taking actions. This would allow a batch update of the model policy.
model_stable = tf.keras.models.clone_model(model)


def preProcess(observation):
    """Reduce the image center to 80x80 frame"""
    cropped_observation = observation[33:-17]
    img_resized = resize(cropped_observation, output_shape=(80, 80), anti_aliasing=True)
    return img_resized


# Frames Memory
frames_state: collections.deque = collections.deque(maxlen=frames_memory_length)
total_frames = 0

max_episodes_tqdm = tqdm.trange(max_episodes)
for episode in max_episodes_tqdm:
    init_state, _ = env.reset()
    frames_state.clear()
    frames_state.append(np.zeros((80, 80)))
    frames_state.append(np.zeros((80, 80)))
    i = 0
    done = False
    episode_reward = 0

    while done != True:
        total_frames += 1
        i += 1
        if i % frames_to_skip != 0:
            continue

        time_series = tf.expand_dims([frames_state[0], frames_state[1]], axis=0)

        # epsilon-greedy selection
        if epsilon > 0 and np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action_probs = model(time_series)
            action = tf.argmax(action_probs[0]).numpy()

        # Take action
        observation, reward, terminated, truncated, info = env.step(action)

        # -Train
        # Predict stable model Q-value
        stable_q_values = model_stable(time_series)
        expected_stable_reward = reward + gamma * tf.reduce_max(stable_q_values, axis=1)

        with tf.GradientTape() as tape:
            # Train the model with the state
            action_probs = model(time_series)
            q_action = tf.reduce_max(action_probs, axis=1)
            loss = model.loss(expected_stable_reward, q_action)

        # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if episode > 1 and episode % 1000 == 0:
            model_stable.set_weights(model.get_weights())

        done = terminated or truncated
        episode_reward += reward
        frames_state.append(preProcess(observation))

        max_episodes_tqdm.set_postfix(
            episode=episode,
            step=i,
            episode_reward=episode_reward,
            epsilon=epsilon,
            total_frames=total_frames,
        )

    # decay epsilon on
    epsilon -= epsilon_decay
    if epsilon < 0.05:
        epsilon = 0


print(f"done: episode_reward={episode_reward}")
