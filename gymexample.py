import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.005
decay_factor = 500
max_episode_steps = 2000
RENDER = False
rewards = []

def linearize(env, i, n_bins):
    return np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=n_bins, endpoint=True)


def discretize(observation, pos, vel):
    return np.digitize(observation[0], pos), np.digitize(observation[1], vel)


def heatmap(agent, episode, n_bins, reward):
    q_vals = agent.action_state_vals
    values = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            values[i][j] = np.max(q_vals[i, j, :])
    plt.imshow(values, cmap='viridis', aspect='auto', extent=[-1.2, 0.6,-0.07, 0.07])
    plt.ylabel('Velocity') # clamped between -1.2 and 0.6 by environment
    plt.xlabel('Position') # clamped between -0.7 and 0.7 by environment
    plt.colorbar()
    plt.title('Heatmap after {} episodes; best reward: {}'.format(episode, reward))
    plt.show()


def reward_plot():
    episodes = [i+1 for i in range(max_episodes[0])]
    plt.plot(episodes, rewards)
    plt.ylabel('Reward') # clamped between -1.2 and 0.6 by environment
    plt.xlabel('Episode') # clamped between -0.7 and 0.7 by environment
    plt.title('Reward over episodes')
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
            return self.policy[discretized_obs]
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
        global rewards
        best_reward = -1000000000
        rewards = []
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
            rewards.append(total)
            if ep % 100 == 0:
                heatmap(self.agent, ep, self.agent.bins, best_reward)
            print('Episode: {}, Reward: {}, Best_reward: {}'.format(ep, total, best_reward))
        reward_plot()
        return np.argmax(self.agent.action_state_vals, axis=2)

if __name__ == "__main__":
    parser = ap.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument('-a',
                        type=float,
                        help='Alpha Value (default = 0.1)')

    parser.add_argument('-g',
                        type=float,
                        help='Gamma value (default = 0.9)')

    parser.add_argument('-e', '--episodes',
                        type=int,
                        help='Maximum number of episodes (default = 50000)')

    parser.add_argument('-r', '--render',
                        type=bool,
                        help='Whether to render the mountain car using the OpenAI Gym environment (default = False)')


    # Execute parse_args()
    args = parser.parse_args()


    # env = gym.make("MountainCar-v0")

    gym.envs.register(id='MountainCarMyEasyVersion-v0', entry_point='gym.envs.classic_control:MountainCarEnv',
                      max_episode_steps=max_episode_steps, )  # MountainCar-v0 uses 200
    env = gym.make('MountainCarMyEasyVersion-v0')

    #max_episodes = np.arange(50000, 100000, 10000, dtype=np.int32)
    #bins = np.arange(50, 1000, 50, dtype=np.int32)
    max_episodes = [10000]
    bins = [50]

    agent = []

    agent.append(QLearning(env, bins[0], max_episodes[0], epsilon))
    agent.append(SARSA(env, bins[0], max_episodes[0], epsilon))

    handler = [Agent(env, agent[i]) for i in range(len(agent))]
    policy = [handler[i].train() for i in range(len(handler))]

    env.close()
