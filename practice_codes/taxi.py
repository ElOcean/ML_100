# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Training parameters for Q learning
alpha = 0.1 # Learning rate
gamma = 0.6 # Future reward discount factor
num_of_episodes = 10000
num_of_steps = 500 # per each episode

# Q tables for rewards
Q_reward = -100000*numpy.ones((500,6)) # All same
#Q_reward = -100000*numpy.random.random((500, 6)) # Random

# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE
# Training w/ random sampling of actions


# Epsilon-greedy parameters
epsilon = 1.0  # Initial exploration probability
min_epsilon = 0.1  # Minimum exploration probability
decay_rate = 0.995  # Decay rate for exploration probability


for episode in range(num_of_episodes):
    #state = env.reset()
    state = env.reset()[0]
    #print("THIS IS THE STATE: ", state)
    for step in range(num_of_steps):
        # Choose an action using an epsilon-greedy strategy
        if random.uniform(0, 1) < min_epsilon:  # 10% chance of choosing a random action
            action = env.action_space.sample()
        else:
            action = numpy.argmax(Q_reward[state])
       # print("GG THIS IS THE ACTION:", action)
        # Take the action and observe the new state and reward
        next_state, reward, done, truncated, info = env.step(action)

        # Update the Q-table
        Q_reward[state, action] = Q_reward[state, action] + alpha * (reward + gamma * numpy.max(Q_reward[next_state]) - Q_reward[state, action])

        # Update the state
        state = next_state
       
        # Break if the episode is done
        if done:
            break
    

# Testing
state = env.reset()[0]
#state, _ = env.reset()
tot_reward = 0
for t in range(50):
    action = numpy.argmax(Q_reward[state,:])
    state, reward, done, truncated, info = env.step(action)
    tot_reward += reward
    print(env.render())
    time.sleep(1)
    if done:
        print("Total reward %d" %tot_reward)
        break