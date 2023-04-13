import collections
import gymnasium as gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt

from typing import List, Tuple

# Hyperparameters
# Small epsilon value for stabilizing division operations
learning_rate = 0.01
gamma = 0.99
max_steps_per_episode = 500
min_episodes_criterion = 100
max_episodes = 20000
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
eps = np.finfo(np.float32).eps.item()

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# `CartPole-v1` is considered solved if average reward is >= 475 over 500 consecutive trials
reward_threshold = -100
running_reward = 0

# Create the environment
env_name = "Acrobot-v1"
render_mode = "rgb_array"
# render_mode="human"
env = gym.make(env_name, render_mode=render_mode)


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int = 128):
        super().__init__()

        self.common = tf.keras.layers.Dense(
            num_hidden_units, activation="relu")
        self.actor = tf.keras.layers.Dense(num_actions, name="actor")
        self.critic = tf.keras.layers.Dense(1, name="critic")

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)


class EnvTfWrap():
    """
    Wrap Gym's `env.step` call as an operation in a TensorFlow function.
    This would allow it to be included in a callable TensorFlow graph.
    """
    @staticmethod
    def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns state, reward and done flag given an action."""

        state, reward, done, truncated, info = env.step(action=action)
        return (state.astype(np.float32),
                np.array(reward, np.int32),
                np.array(done, np.int32))

    @staticmethod
    def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(EnvTfWrap.env_step, [action], [tf.float32, tf.int32, tf.int32])


class TrainingProcess():
    def __init__(
            self,
            model: tf.keras.Model):
        self.model = model

    def run_episode(
            self,
            initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps_per_episode):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)

            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply action to the environment to get next state and reward
            state, reward, done = EnvTfWrap.tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    def get_expected_return(
            self,
            rewards: tf.Tensor,
            standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) /
                       (tf.math.reduce_std(returns) + eps))

        return returns

    def compute_loss(
            self,
            action_probs: tf.Tensor,
            values: tf.Tensor,
            returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined Actor-Critic loss."""

        advantage = returns - values
        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
        critic_loss = huber_loss(values, returns)

        return actor_loss + critic_loss

    @ tf.function
    def train_step(
            self,
            initial_state: tf.Tensor) -> tf.Tensor:
        """Runs a model training step."""

        with tf.GradientTape() as tape:

            # Run the model for one episode to collect training data
            action_probs, values, rewards = self.run_episode(initial_state)

            # Calculate the expected returns
            returns = self.get_expected_return(rewards)

            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # Calculate the loss values to update our network
            loss = self.compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        # episode_reward
        return tf.math.reduce_sum(rewards)


# Constract the trainins
model = ActorCritic(env.action_space.n)
trainingProcess = TrainingProcess(model)

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(
    maxlen=min_episodes_criterion)

episodes_reward_stats = []
print(
    f'Training started for: {env_name}, target: {reward_threshold}, (last {min_episodes_criterion} runs).')
t = tqdm.trange(max_episodes)
for i in t:
    initial_state, info = env.reset()
    initial_state = tf.constant(initial_state, dtype=tf.float32)
    episode_reward = int(trainingProcess.train_step(initial_state))

    episodes_reward.append(episode_reward)
    episodes_reward_stats.append(episode_reward)
    running_reward = statistics.mean(episodes_reward)

    # Show the average episode reward every 10 episodes
    if i % 10 == 0:
        t.set_postfix(episode_reward=episode_reward,
                      avrange_running_reward=running_reward)

    if running_reward > reward_threshold and i >= min_episodes_criterion:
        break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
# Solved at episode 3052: average reward: -98.72!

steps = range(0, i+1, 1)
plt.plot(steps, episodes_reward_stats)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.ylim(top=max(episodes_reward_stats)*1.1)
plt.show()
