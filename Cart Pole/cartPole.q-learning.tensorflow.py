import gymnasium as gym
import numpy as np
import tensorflow as tf
import tqdm

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# Hyperparameters
seed = 42
learning_rate = 0.01
discount_factor = 0.99
num_episodes = 5000
optimizer = tf.keras.optimizers.Adam(learning_rate)
reward_threshold = 475
exploration_rate = 0.1

# Define the environment and set up the neural network
render_mode = "rgb_array"
# render_mode="human"
env = gym.make("CartPole-v1", render_mode=render_mode)
env_shape = env.observation_space.shape

model = tf.keras.Sequential([
    tf.keras.Input(shape=env_shape),
    tf.keras.layers.Dense(128, name="input-layer-act2", activation='relu'),
    tf.keras.layers.Dense(
        env.action_space.n, activation='softmax', name="output-layer")
])


def calcActionProbabilities(state: list[float, float, float, float]):
    return model(tf.convert_to_tensor([state], dtype=tf.float32))


@tf.function
def train(state, next_state, reward, done):
    with tf.GradientTape() as tape:
        # Compute the Q-values for the current state and action
        q_value = calcActionProbabilities(state)

        # Compute the target Q-values for the next state
        next_q_value = calcActionProbabilities(next_state)
        max_future_q_value = tf.reduce_max(next_q_value, axis=1)
        target = reward + discount_factor * max_future_q_value * (1 - done)

        # Compute the loss and gradients
        loss = tf.reduce_mean(tf.square(q_value - target))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))
    return loss

# Define the training loop and implement the Q-learning algorithm


training_rewards = []

t = tqdm.trange(num_episodes)
for i in t:
    state, _ = env.reset()
    done = False
    total_reward = 0

    # Take action and observe next state and reward
    while not done:
        # Choose action using exploration_rate or policy
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = calcActionProbabilities(state)
            action = tf.argmax(action[0]).numpy()

        next_state, reward, done, truncated, _ = env.step(action)

        loss = train(state=state, reward=reward,
                     done=done, next_state=next_state)
        # Update variables
        state = next_state
        total_reward += reward

    if i % 10 == 0 and (len(training_rewards) > 0):
        t.set_postfix(total_reward_averange=np.mean(
            training_rewards), loss=loss)

    training_rewards.append(total_reward)

print(f"\nAverage training reward = {np.mean(training_rewards)}\n")

# Test the model on the environment
num_test_episodes = 10
test_rewards = []

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action_probs = calcActionProbabilities(state)
        action = tf.argmax(action_probs[0]).numpy()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    test_rewards.append(total_reward)

print(f"\nAverage test reward = {np.mean(test_rewards)}\n")
