import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

alpha = 0.05
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.005
decay_factor = 500
max_episode_steps = 200

class QLearning:
    def __init__(self, env, bins, max_episodes):
        global alpha, gamma, epsilon, epsilon_min
        self.env = env
        self.decay = decay_factor * epsilon_min / (max_episodes * max_episode_steps)


        self.pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=bins, endpoint=True)
        self.vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=bins, endpoint=True)
        self.action_state_vals = np.zeros((bins+1, bins+1, env.action_space.n))

    def discretize(self, observation):
        return np.digitize(observation[0], self.pos), np.digitize(observation[1], self.vel)

    def select(self, observation):
        discretized_obs = self.digitize(observation)



class SARSA:
    def __init__(self, env, bins, max_episodes):
        self.env = env



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
