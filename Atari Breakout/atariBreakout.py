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
max_episodes = 5
epsilon = 1
epsilon_decay = 1 / max_episodes
epsilon_random_episodes_len = 2

# Hyperparameters (env)
repeat_action_probability = 0
# render_mode = "human"
render_mode = "rgb_array"
obs_type = "grayscale"
# Make env
env = gym.make(
    "ALE/Breakout-v5",
    render_mode=render_mode,
    repeat_action_probability=repeat_action_probability,
    obs_type=obs_type,
)

# Sequenctial-visual model, insert 2 frames at every step
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
model.compile()
model.summary()

# Training model used for training, and not for taking actions. This would allow a batch update of the model policy.
training_model = tf.keras.models.clone_model(model)


def preProcess(observation):
    cropped_observation = observation[33:-17]
    img_resized = resize(cropped_observation, output_shape=(80, 80), anti_aliasing=True)
    return img_resized


# # Initialize the Gym environment
frames_state: collections.deque = collections.deque(maxlen=frames_memory_length)

max_episodes_tqdm = tqdm.trange(max_episodes)
for episode in max_episodes_tqdm:
    init_state, _ = env.reset()
    frames_state.append(np.zeros((80, 80)))
    i = 0
    done = False
    episode_reward = 0

    while done != True:
        i += 1
        if i % frames_to_skip != 0:
            continue

        # epsilon greedy selection
        if epsilon > 0 and np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action_probs = model(time_series)
            action = tf.argmax(action_probs[0]).numpy()
        # Take action
        observation, reward, terminated, truncated, info = env.step(action)

        # Train
        done = terminated or truncated
        episode_reward += reward
        next_state = preProcess(observation)
        frames_state.append(next_state)
        time_series = tf.expand_dims(frames_state, axis=0)

    max_episodes_tqdm.set_postfix(epsilon=epsilon)

    # decay epsilon on
    epsilon -= epsilon_decay
    if epsilon < 0.05:
        epsilon = 0


print(terminated, truncated, info)
print(f"done: episode_reward={episode_reward}")
plt.imshow(next_state, cmap="gray")
plt.axis("on")
plt.show()
plt.close()
