import tensorflow as tf

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from skimage.transform import resize

# Hyperparameters
filters = 32
actions = 4
kernel_size = (4, 4)
strides = (2, 2)
frames_to_skip = 15

# Hyperparameters (env)
repeat_action_probability = 0
render_mode = "rgb_array"
obs_type = "grayscale"
# Make env
env = gym.make("ALE/Breakout-v5", render_mode=render_mode,
               repeat_action_probability=repeat_action_probability, obs_type=obs_type)

# Sequenctial-visual model, insert 2 frames at every step
model = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D(filters, kernel_size=4, input_shape=(
        2, 80, 80, 1), padding='same', strides=4),
    tf.keras.layers.Conv2D(filters, kernel_size=2, strides=2),
    tf.keras.layers.Conv2D(filters*2, kernel_size=2, strides=1),
    tf.keras.layers.Dense(filters*2, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(
        actions, activation=tf.keras.activations.softmax, name="output-layer")
])
model.compile()
model.summary()


def preProcess(observation):
    cropped_observation = observation[33:-17]
    img_resized = resize(cropped_observation,
                         output_shape=(80, 80), anti_aliasing=True)
    return img_resized


# # Initialize the Gym environment
init_state, _ = env.reset()
prev_state = np.zeros((80, 80))
action = 1

i = 1
done = False
episode_reward = 0

while done != True:
    if i % frames_to_skip != 0:
        i += 1
        continue
    i = 1

    # Take action
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward
    next_state = preProcess(observation)
    time_series = np.expand_dims([prev_state, next_state], axis=0)

    # epsilon greedy selection
    action_probs = model(time_series)
    action = np.argmax(action_probs)
    action = 1

    # Train

    # prepare for next frame
    prev_state = next_state


print(terminated, truncated, info)
print(f"done: episode_reward={episode_reward}")
plt.imshow(next_state, cmap='gray')
plt.axis('on')
plt.show()
plt.close()
