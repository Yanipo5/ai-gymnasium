import gymnasium as gym
import numpy as np
import tensorflow as tf

# Define the environment and set up the neural network
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1", render_mode="rgb_array")
num_actions = env.action_space.n
input_shape = env.observation_space.shape[0]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax')
])

# Define the training loop and implement the Q-learning algorithm
learning_rate = 0.05
discount_factor = 0.95
num_episodes = 50
epsilon = 0.15

optimizer = tf.keras.optimizers.Adam(learning_rate)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_array = state[0]
            state_tensor = tf.convert_to_tensor(state_array[np.newaxis], dtype=tf.float32)
            action_probs = model(state_tensor)
            action = tf.argmax(action_probs[0]).numpy()
        
        # Take action and observe next state and reward
        next_state, reward, done, info, dict = env.step(action)

        # Update Q-values using Q-learning algorithm
        with tf.GradientTape() as tape:
            state_array = np.array(state[0])
            state_tensor = tf.convert_to_tensor(state_array[np.newaxis], dtype=tf.float32)
            action_probs = model(state_tensor)
            q_values = tf.reduce_sum(action_probs * model.weights[-1], axis=1)
            next_state_array = np.array(next_state)
            next_state_tensor = tf.convert_to_tensor(next_state_array[np.newaxis], dtype=tf.float32)
            max_q_value = tf.reduce_max(model(next_state_tensor), axis=1)
            target_q = reward + discount_factor * max_q_value * (1 - int(done))
            loss = tf.reduce_mean(tf.square(q_values - target_q))


        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update variables
        # print(state)
        # print([next_state,all])
        state = [next_state,dict]
        total_reward += reward

    print(f"Episode {episode}: Total reward = {total_reward}")

# Test the model on the environment
num_test_episodes = 30
test_rewards = []

for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_array = state[0]
        state_tensor = tf.convert_to_tensor(state_array[np.newaxis], dtype=tf.float32)
        action_probs = model(state_tensor)
        action = tf.argmax(action_probs[0]).numpy()
        next_state, reward, done, info, dict = env.step(action)
        state = [next_state,dict]
        total_reward += reward

    test_rewards.append(total_reward)
    print(f"Test episode {episode}: Total reward = {total_reward}")

print(f"Average test reward = {np.mean(test_rewards)}")
