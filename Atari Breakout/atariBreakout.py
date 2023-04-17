import tensorflow as tf
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from skimage.transform import resize

# Hyperparameters
filters = 32
filters2 = int(filters/2)
action_space = (4,)
kernel_size = (2, 2)
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
    tf.keras.layers.ConvLSTM2D(filters2, kernel_size=(4, 4), input_shape=(
        None, 80, 80, 1), padding='same', return_sequences=True, strides=(4, 4)),
    tf.keras.layers.ConvLSTM2D(filters, kernel_size=(
        4, 4), padding='same', return_sequences=True, strides=(4, 4)),
    tf.keras.layers.ConvLSTM2D(filters2, kernel_size=(
        4, 4), padding='same', return_sequences=False, strides=(4, 4)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax', name="output-layer")
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


i = 0
f = 0
while True:
    i += 1
    if i % frames_to_skip != 0:
        continue

    f += 1
    observation, reward, terminated, truncated, info = env.step(action)
    if f >= 10:
        cropped_observation = observation[33:-17]
        img_resized = resize(cropped_observation,
                             output_shape=(80, 80), anti_aliasing=True)

        plt.imshow(img_resized, cmap='gray')
        plt.axis('on')
        plt.show()
        print(img_resized.shape)
        break
