import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse as ap

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.005
decay_factor = 500
max_episode_steps = 10000
RENDER = False


def linearize(env, i, n_bins):
    return np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=n_bins, endpoint=True)


def discretize(observation, pos, vel):
    return np.digitize(observation[0], pos), np.digitize(observation[1], vel)


def heatmap(agent, episode, n_bins):
    q_vals = agent.action_state_vals
    values = np.zeros((n_bins+1, n_bins+1))
    for i in range(n_bins+1):
        for j in range(n_bins+1):
            values[i][j] = np.argmax(q_vals[i, j, :])
    plt.imshow(values, cmap='viridis', aspect='auto', extent=[-0.07, 0.07, -1.2, 0.6])
    plt.ylabel('Position') # clamped between -1.2 and 0.6 by environment
    plt.xlabel('Velocity') # clamped between -0.7 and 0.7 by environment
    plt.colorbar()
    plt.title('Heatmap after {} episodes'.format(episode))
    plt.show()


class QLearning:
    def __init__(self, env, bins, max_episodes, eps):
        global alpha, gamma, epsilon, epsilon_min
        self.env = env
        self.bins = bins
        self.max_eps = max_episodes
        self.decay = decay_factor * epsilon_min / (max_episodes * max_episode_steps)
        self.epsilon = eps

        self.pos = linearize(env, 0, bins)  # from -1.2 to 0.6 for bins
        self.vel = linearize(env, 1, bins)  # from -0.07 to 0.07 for bins
        self.action_state_vals = np.zeros((bins + 1, bins + 1, env.action_space.n))

    def select(self, observation):
        discretized_obs = discretize(observation, self.pos, self.vel)
        if self.epsilon > epsilon_min:
            self.epsilon -= self.decay
        if np.random.random() > self.epsilon:
            return np.argmax(self.action_state_vals[discretized_obs])
        else:
            return np.random.choice(env.action_space.n)

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = discretize(obs, self.pos, self.vel)
        discretized_next_obs = discretize(next_obs, self.pos, self.vel)
        action_values = reward + gamma * np.max(self.action_state_vals[discretized_next_obs]) - \
                        self.action_state_vals[discretized_obs][action]
        self.action_state_vals[discretized_obs][action] += alpha * action_values


class SARSA:
    def __init__(self, env, bins, max_episodes, eps):
        global alpha, gamma, epsilon, epsilon_min
        self.env = env
        self.bins = bins
        self.max_eps = max_episodes
        self.decay = decay_factor * epsilon_min / (max_episodes * max_episode_steps)
        self.epsilon = eps

        self.pos = linearize(env, 0, bins)  # from -1.2 to 0.6 for bins
        self.vel = linearize(env, 1, bins)  # from -0.07 to 0.07 for bins
        self.action_state_vals = np.zeros((bins + 1, bins + 1, env.action_space.n))
        self.policy = np.random.randint(0, env.action_space.n, size=(bins + 1, bins + 1))

    def select(self, observation):
        discretized_obs = discretize(observation, self.pos, self.vel)
        if self.epsilon > epsilon_min:
            self.epsilon -= self.decay
        if np.random.random() > self.epsilon:
            return np.policy[discretized_obs]
        else:
            return np.random.choice(env.action_space.n)

    def learn(self, obs, action, reward, next_obs):
        discretized_obs = discretize(obs, self.pos, self.vel)
        discretized_next_obs = discretize(next_obs, self.pos, self.vel)
        next_action = self.policy[discretized_next_obs]
        action_values = reward + gamma * self.action_state_vals[discretized_next_obs][next_action] - \
                        self.action_state_vals[discretized_obs][action]
        self.action_state_vals[discretized_obs][action] += alpha * action_values
        self.policy[discretized_obs] = np.argmax(self.action_state_vals[discretized_obs])


class Agent:
    def __init__(self, env, agent):
        self.agent = agent
        self.env = env

    def train(self):
        best_reward = -1000000000
        for ep in range(1, self.agent.max_eps + 1):
            finished = False
            total = 0.0
            obs = self.agent.env.reset()
            while not finished:
                action = self.agent.select(obs)
                next_obs, reward, finished, info = self.agent.env.step(action)
                self.agent.learn(obs, action, reward, next_obs)
                obs = next_obs
                total += reward
                if reward > best_reward and RENDER:
                    self.env.render()
            best_reward = max(best_reward, total)
            if ep % 50 == 0:
                heatmap(self.agent, ep, self.agent.bins)
            print('Episode: {}, Reward: {}, Best_reward: {}'.format(ep, total, best_reward))
        return np.argmax(self.agent.action_state_vals, axis=2)

    def test(self, policy):
        finished = False
        total = 0.0
        obs = self.env.reset()
        while not finished:
            action = policy[discretize(obs, self.agent.pos, self.agent.vel)]
            next_obs, reward, done, info = self.env.step(action)
            obs = next_obs
            total += reward
        return total


if __name__ == "__main__":
    slow = True

    # env = gym.make("MountainCar-v0")

    gym.envs.register(id='MountainCarMyEasyVersion-v0', entry_point='gym.envs.classic_control:MountainCarEnv',
                      max_episode_steps=max_episode_steps, )  # MountainCar-v0 uses 200
    env = gym.make('MountainCarMyEasyVersion-v0')

    #max_episodes = np.arange(50000, 100000, 10000, dtype=np.int32)
    #bins = np.arange(50, 1000, 50, dtype=np.int32)
    max_episodes = [50000]
    bins = [50]

    agent = []

    agent.append(QLearning(env, bins[0], max_episodes[0], epsilon))
    agent.append(SARSA(env, bins[0], max_episodes[0], epsilon))

    handler = [Agent(env, agent[i]) for i in range(len(agent))]
    policy = [handler[i].train() for i in range(len(handler))]

    # output_dir = './q_output'
    # env = gym.wrappers.Monitor(env, output_dir, force=True)
    for i in range(len(handler)):
        for _ in range(1000):
            env.render()
            [handler[i].test(policy)]

    env.close()

"""
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
"""
