import gym
import numpy as np

env = gym.make('CartPole-v1')

env.reset()

def run_episode(params):
    obs = env.reset()
    total_reward = 0
    for t in range(200):
        action = 0 if np.matmul(params, obs) < 0 else 1
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

params = np.random.rand(4)
bestreward = 0
step = 0
step_scaling = 0.25


for _ in range(10000):
    step += 1
    new_params = params + np.random.rand(4) * step_scaling 
    reward = run_episode(new_params)
    if reward > bestreward:
        params = new_params
        bestreward = reward
        if reward == 200:
            break

print(bestreward)
print(step)

            
