import gymnasium as gym
import numpy as np
import tensorflow as tf
import tqdm

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")

# Define the environment and set up the neural network
render_mode = "rgb_array"
# render_mode="human"
env = gym.make("CartPole-v1", render_mode=render_mode)
action_space = env.action_space.n
env_shape = env.observation_space.shape

model = tf.keras.Sequential([
    tf.keras.Input(shape=env_shape),
    tf.keras.layers.Dense(128, name="input-layer-act", activation='relu'),
    tf.keras.layers.Dense(64, name="input-layer-act-2"),
    tf.keras.layers.Dense(
        action_space, activation='softmax', name="output-layer")
])


def calcActionProbabilities(state_array: list[float]):
    state_tensor = tf.convert_to_tensor(
        [state_array], dtype=tf.float32)
    return model(state_tensor)


@tf.function
def train(state, model, next_state, reward, done):
    # Update Q-values using Q-learning algorithm
    with tf.GradientTape() as tape:
        action_probs = calcActionProbabilities(state)
        q_values = tf.reduce_sum(action_probs * model.weights[-1], axis=1)
        next_state_tensor = tf.convert_to_tensor(
            [next_state], dtype=tf.float32)
        max_q_value = tf.reduce_max(model(next_state_tensor), axis=1)
        target_q = reward + discount_factor * max_q_value * (1 - int(done))
        loss = tf.reduce_mean(tf.square(q_values - target_q))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# Define the training loop and implement the Q-learning algorithm
learning_rate = 0.01
discount_factor = 0.99
num_episodes = 500
exploration_rate = 0.1
training_rewards = []

optimizer = tf.keras.optimizers.Adam(learning_rate)

t = tqdm.trange(num_episodes)
for i in t:
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_probs = None
        env.render()
        # Choose action using exploration_rate or policy
        if np.random.rand() < exploration_rate:
            action = env.action_space.sample()
        else:
            action_probs = calcActionProbabilities(state)
            action = tf.argmax(action_probs[0]).numpy()

        # Take action and observe next state and reward
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        train(state=state, model=model, reward=reward,
              done=done, next_state=next_state)
        # Update variables
        state = next_state
        total_reward += reward

    if (len(training_rewards) > 0):
        t.set_postfix(total_reward_averange=np.mean(training_rewards))
    if i % 10 == 0:
        pass
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
