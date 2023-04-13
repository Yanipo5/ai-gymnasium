import gymnasium as gym
import statistics
import numpy as np
import tqdm

import tensorflow as tf
import tf_agents as tf_agents
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import time_step

# TEST
print("=======================")
print("CartPole-v1 RL Started:")

# Hyperparameters
seed = 42  # @param {type:"integer"}
max_training_iterations = 20000  # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
discount_factor = 0.99


# Create the environment
render_mode = "rgb_array"
# render_mode="human"
env = gym.make("CartPole-v1", render_mode=render_mode)

observation_spac = BoundedTensorSpec(
    shape=env.observation_space.shape, dtype=tf.float32, minimum=env.observation_space.low, maximum=env.observation_space.high, name="observation")
action_spec = BoundedTensorSpec(
    shape=env.action_space.shape, dtype=tf.int32, minimum=0, maximum=env.action_space.n-1, name='action')

q_net = q_network.QNetwork(
    input_tensor_spec=observation_spac,
    action_spec=action_spec,
)

agent = dqn_agent.DqnAgent(
    time_step_spec=time_step.TimeStep(
        discount=BoundedTensorSpec(
            shape=(), dtype=tf.float32, minimum=0, maximum=discount_factor, name="discount"),
        observation=observation_spac,
        reward=TensorSpec(shape=(), dtype=tf.int32, name='reward'),
        step_type=TensorSpec(shape=(), dtype=tf.int32, name='step_type')
    ),
    action_spec=action_spec,
    q_network=q_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate),
)
agent.initialize()


# Start training process
# t = tqdm.trange(max_training_iterations)
running_reward = []
t = tqdm.trange(1)
for i in t:
    done = False
    step_type = tf_agents.trajectories.StepType.FIRST
    reward = 1
    episode_reward = reward
    observation, _ = env.reset(seed=seed)
    observation = tf.constant(observation, dtype=tf.float32)
    observation = tf.expand_dims(observation, 0)

    policy_step = agent.policy.action(time_step.TimeStep(
        observation=tf.convert_to_tensor([observation], dtype=tf.float32),
        discount=tf.constant(discount_factor, dtype=tf.float32),
        reward=tf.constant(reward, dtype=tf.int32),
        step_type=tf.constant(
            tf_agents.trajectories.StepType.FIRST, dtype=tf.int32)
    ))
    action = policy_step.action.numpy()[0][0]
    step_type = tf_agents.trajectories.StepType.MID

    while not done:
        state, reward, terminated, truncated, info = env.step(action)
        reward = int(reward)
        episode_reward += reward
        done = terminated or truncated
        if done:
            step_type = tf_agents.trajectories.StepType.LAST

        agent.train(tf_agents.trajectories.Trajectory(
            step_type=tf.constant(step_type, dtype=tf.int32),
            observation=tf.convert_to_tensor(observation, dtype=tf.float32),
            action=tf.constant(action, dtype=tf.int32),
            reward=tf.constant(reward, dtype=tf.int32),
            discount=tf.constant(discount_factor, dtype=tf.float32),
            policy_info=policy_step.info,
            next_step_type=tf.constant(step_type, dtype=tf.int32),
        ))
        observation = tf.constant(state, dtype=tf.float32)
        observation = tf.expand_dims(observation, 0)
        running_reward.append(episode_reward)

    t.set_postfix(
        episode_reward=episode_reward, avg_rewards=statistics.mean(running_reward))


# agent.train(trajectories)
print("CartPole-v1 RL Ended:")
