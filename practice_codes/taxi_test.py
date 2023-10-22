import gym
import numpy as np
import random
import time

# Create the Taxi-V3 environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Initialize the Q-table
#q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table = -1*np.ones((500,6)) # All same
#q_table = -100000*np.random.random((500, 6)) # Random

#print(q_table)


# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode


def train_agent():
    epsilon = 1.0  # Exploration rate
    min_epsilon = 0.01  # Minimum exploration rate
    decay_rate = 0.99  # Decay rate for exploration

    for episode in range(num_of_episodes):
        state = env.reset()[0]  # Reset the environment for a new episode
        done = False  # Initialize done flag for episode termination

        for step in range(num_of_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Choose a random action
                #print("RANDOM ACTION")
            else:
                action = np.argmax(q_table[state])  # Choose the action with the highest Q-value
                #print("Q_VALUE ACTION")

            next_state, reward, done,truncated, info = env.step(action)
            # Q-learning
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            
            state = next_state

            # If the episode is done, break out of the loop
            if done:
                break

          # ------------------ Commented out to keep same exploration  ---------------------- #      
        # Decay the exploration rate
        #epsilon = max(min_epsilon, epsilon * decay_rate)
       
        
train_agent()


cumulative_reward = 0
cumulative_actions = 0

# Run the testing 10 times
for i in range(10):

    # Testing
    state = env.reset()[0]
    tot_reward = 0
    num_actions = 0  # Initialize the number of actions taken

    for t in range(50):
        action = np.argmax(q_table[state, :])
        state, reward, done, truncated, info = env.step(action)
        tot_reward += reward
        num_actions += 1  # Increment the number of actions taken

        #print("REWARD FROM THE ACTION: ", reward)
        #print("CUMULATIVE REWARD: ", tot_reward)
        #print("NUM OF ACTIONS: ", num_actions)
        print(env.render())
        time.sleep(0.5)

        if done:
            print(f"Total reward: {tot_reward}")
            print(f"Number of actions: {num_actions}")
            break
    
    cumulative_reward += tot_reward
    cumulative_actions += num_actions

# print avg reward and actions in 10 runs
print(f"Total average reward: {cumulative_reward/10}")
print(f"Average number of actions: {cumulative_actions/10}")
