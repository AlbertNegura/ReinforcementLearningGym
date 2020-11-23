import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    slow = True

    env = gym.make("MountainCar-v0")

    #gym.envs.register(id='MountainCarMyEasyVersion-v0',entry_point='gym.envs.classic_control:MountainCarEnv',max_episode_steps=100000,)  # MountainCar-v0 uses 200
    #env = gym.make('MountainCarMyEasyVersion-v0')

    action_sapce = env.action_space
    epsilon = 1
    learning_rate = 0.001
    discount_rate = 0.999

    Q_values = np.zeros((20, 20, action_sapce.n))
    max_episode_steps = 10000
    env._max_episode_steps = max_episode_steps


    for _ in range(10):
        observation = env.reset()
        done = False
        timesteps = 0
        while not done:
            if slow: env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            timesteps += 1
            if slow: print(observation)
            if slow: print(reward)
            if slow: print(done)
        print(f"Episode finished after {timesteps} timesteps.")
