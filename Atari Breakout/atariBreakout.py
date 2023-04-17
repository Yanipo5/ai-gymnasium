import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from skimage.transform import resize

# Hyperparameters
filters = 64
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

model = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D(filters, kernel_size=kernel_size, input_shape=(
        None, 80, 80, 1), padding='same', strides=kernel_size, stateful=True, batch_size=1),
    tf.keras.layers.Dense(filters, activation='relu'),
    tf.keras.layers.Dense(actions, activation='softmax', name="output-layer")
])
model.compile()
model.summary()

# # Initialize the Gym environment
init_state, _ = env.reset()
prev_state = np.zeros((80, 80, 1))
prev_state = np.expand_dims(prev_state, axis=0)
prev_state = np.expand_dims(prev_state, axis=0)
action_probs = model(prev_state)
action = np.argmax(action_probs)
action = 1


def preProcess(observation):
    cropped_observation = observation[33:-17]
    img_resized = resize(cropped_observation,
                         output_shape=(80, 80), anti_aliasing=True)
    return img_resized


i = 1
done = False
episode_reward = 0

while done != True:
    if i % frames_to_skip != 0:
        i += 1
        continue
    i = 1

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    episode_reward += reward

    img_pre_processed = preProcess(observation)

    img_resized = np.expand_dims(img_pre_processed, axis=0)
    img_resized = np.expand_dims(img_resized, axis=0)
    action_probs = model(img_resized)
    action = np.argmax(action_probs)
    action = 1

print(terminated, truncated, info)
print(f"done: episode_reward={episode_reward}")
plt.imshow(img_pre_processed, cmap='gray')
plt.axis('on')
plt.show()
plt.close()
