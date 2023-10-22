import gym
import numpy as np
import random

# Create the Taxi-V3 environment
env = gym.make('Taxi-v3',render_mode='ansi')

# Initialize the Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.995  # Decay rate for exploration

# Training parameters
num_episodes = 50000  # Number of episodes for training (increased from 10000 to 50000)
max_steps_per_episode = 100  # Maximum steps per episode

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]  # Reset the environment for a new episode
    done = False  # Initialize done flag for episode termination
    total_reward = 0  # Initialize total reward for this episode

    for step in range(max_steps_per_episode):
        # Epsilon-greedy strategy for action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Choose a random action
        else:
            action = np.argmax(q_table[state, :])  # Choose the action with the highest Q-value

        # Execute the action in the environment
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Accumulate the reward

        # Update the Q-value using the Q-learning update rule
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

        # Update the state for the next iteration
        state = next_state

        # If the episode is done, break out of the loop
        if done:
            break

    # Decay the exploration rate
    epsilon = max(min_epsilon, epsilon * decay_rate)

# ... (rest of your code for evaluation)

# Evaluation parameters
num_evaluation_episodes = 10  # Number of episodes for evaluation
total_evaluation_reward = 0  # Initialize total evaluation reward

# Evaluation loop
for episode in range(num_evaluation_episodes):
    state = env.reset()[0]  # Reset the environment for a new episode
    done = False  # Initialize done flag for episode termination

    while not done:
        action = np.argmax(q_table[state, :])  # Choose the action with the highest Q-value
        next_state, reward, done, truncated, info = env.step(action)  # Execute the action in the environment
        total_evaluation_reward += reward  # Accumulate the reward
        state = next_state  # Update the state for the next iteration

# Output the average evaluation reward
average_evaluation_reward = total_evaluation_reward / num_evaluation_episodes
print(f'Average evaluation reward: {average_evaluation_reward}')
