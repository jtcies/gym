# code from:
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

import gym
import numpy as np
import scipy
import random

env = gym.make('Roulette-v0').env

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# hyperpamaters
alpha = 0.1
epsilon = 0.1
gamma = 0.6

# collect the data
all_epochs = []
all_penalites = []

n_runs = 100000

for _ in range(n_runs):
    state = env.reset()
    epochs, penalites, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else: 
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # q table formula
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalites += 1

        state = next_state

        epochs += 1

print(epochs)
print(penalites)

## evalute agent

total_epochs, total_penalites = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalites, rewards = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalites += 1

        epochs += 1


    total_penalites += penalites
    total_epochs += epochs

print(total_epochs / episodes)
print(total_penalites / episodes)




