"""
Authors: Viktor Cef Inselberg (i6157970), Albert Negura (i6145864)
Reinforcement learning using OpenAI Gym and the Mountaincar environment.
Specifically, SARSA and Q-Learning are implemented.
Visualization methods are also included.
Please note the argparse if running from console (explanation included in README).
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

alpha = 0.9
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.005
decay_factor = 500
max_episode_steps = 2000
RENDER = False
rewards = []
DEBUG = True
CONSTANT = True

def linearize(env, i, n_bins):
    """
    Generate a linear array between the lowest point in the observation space and highest point in the observation space of the environment.
    :param env: the given environment
    :param i: the observation number (0 position, 1 velocity)
    :param n_bins: the number of values to generate
    :return: a linearly spaced vector
    """
    return np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=n_bins, endpoint=True)


def discretize(observation, pos, vel):
    """
    Discretize a given observation space according to the given linearly-spaced vectors.
    :param observation: the observation space
    :param pos: the position vector
    :param vel: the velocity vector
    :return: a discretized version of the space.
    """
    return np.digitize(observation[0], pos), np.digitize(observation[1], vel)


def heatmap(agent, episode, n_bins, reward):
    """
    Generate a heatmap of the velocity against position.
    :param agent: the agent to obtain the action state values from (for the velocity and position values).
    :param episode: which episode the heatmap is being generated for
    :param n_bins: resolution of the heatmap (square)
    :param reward: the best reward obtained so far
    :return: nothing
    """
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
    """
    Generate a linear plot of the reward over all episodes.
    :return: nothing
    """
    episodes = [i+1 for i in range(max_episodes[0])]
    plt.plot(episodes, rewards)
    plt.ylabel('Reward') # clamped between -1.2 and 0.6 by environment
    plt.xlabel('Episode') # clamped between -0.7 and 0.7 by environment
    plt.title('Reward over episodes')
    plt.show()


class QLearning:
    def __init__(self, env, bins, max_episodes, eps):
        """
        Initialize a Q-Learning agent
        :param env: the environment the agent is being trained on
        :param bins: the dimension of the discretized version of the space
        :param max_episodes: maximum number of episodes the agent will be trained on
        :param eps: initial value for epsilon for the epsilon greedy selection strategy
        """
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
        """
        Select an value according to an epsilon greedy selection strategy.
        :param observation: the current observation space
        :return: the selected action according to epsilon greedy (either "best" action or random)
        """
        discretized_obs = discretize(observation, self.pos, self.vel)
        if self.epsilon > epsilon_min:
            self.epsilon -= self.decay
        if np.random.random() > self.epsilon:
            return np.argmax(self.action_state_vals[discretized_obs])
        else:
            return np.random.choice(env.action_space.n)

    def learn(self, obs, action, reward, next_obs):
        """
        Update the action state values of the agent for the given time step according to the agent's update method.
        :param obs: the current observation
        :param action: the current action
        :param reward: the reward for the current action
        :param next_obs: the next observation
        :return: nothing
        """
        discretized_obs = discretize(obs, self.pos, self.vel)
        discretized_next_obs = discretize(next_obs, self.pos, self.vel)
        action_values = reward + gamma * np.max(self.action_state_vals[discretized_next_obs]) - \
                        self.action_state_vals[discretized_obs][action]
        self.action_state_vals[discretized_obs][action] += alpha * action_values


class SARSA:
    def __init__(self, env, bins, max_episodes, eps):
        """
        Initialize a SARSA agent
        :param env: the environment the agent is being trained on
        :param bins: the dimension of the discretized version of the space
        :param max_episodes: maximum number of episodes the agent will be trained on
        :param eps: initial value for epsilon for the epsilon greedy selection strategy
        """
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
        """
        Select an value according to an epsilon greedy selection strategy.
        :param observation: the current observation space
        :return: the selected action according to epsilon greedy (either "best" action or random)
        """
        discretized_obs = discretize(observation, self.pos, self.vel)
        if self.epsilon > epsilon_min:
            self.epsilon -= self.decay
        if np.random.random() > self.epsilon:
            return self.policy[discretized_obs]
        else:
            return np.random.choice(env.action_space.n)

    def learn(self, obs, action, reward, next_obs):
        """
        Update the action state values of the agent for the given time step according to the agent's update method.
        :param obs: the current observation
        :param action: the current action
        :param reward: the reward for the current action
        :param next_obs: the next observation
        :return: nothing
        """
        discretized_obs = discretize(obs, self.pos, self.vel)
        discretized_next_obs = discretize(next_obs, self.pos, self.vel)
        next_action = self.policy[discretized_next_obs]
        action_values = reward + gamma * self.action_state_vals[discretized_next_obs][next_action] - \
                        self.action_state_vals[discretized_obs][action]
        self.action_state_vals[discretized_obs][action] += alpha * action_values
        self.policy[discretized_obs] = np.argmax(self.action_state_vals[discretized_obs])


class Agent:
    def __init__(self, env, agent):
        """
        Initialize an agent handler to handle the training of the agent.
        :param env: the environment to train the agent in.
        :param agent: the agent to be trained.
        """
        self.agent = agent
        self.env = env

    def train(self):
        """
        Train the agent and plot the results.
        :return: the trained policy
        """
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
            if DEBUG:
                if ep in [100, 300, 500, 700, 1000, 2000, 2500, 5000, 10000]:
                    heatmap(self.agent, ep, self.agent.bins, best_reward)
            if RENDER or CONSTANT:
                heatmap(self.agent, ep, self.agent.bins, best_reward)
            print('Episode: {}, Reward: {}, Best_reward: {}'.format(ep, total, best_reward))
        reward_plot()
        return np.argmax(self.agent.action_state_vals, axis=2)

if __name__ == "__main__":
    parser = ap.ArgumentParser()

    parser.add_argument('-a',
                        type=float,
                        default=0.1,
                        help='Alpha Value (default = 0.1)')

    parser.add_argument('-g',
                        type=float,
                        default=0.9,
                        help='Gamma value (default = 0.9)')

    parser.add_argument('-e', '--episodes',
                        type=int,
                        default=10000,
                        help='Maximum number of episodes (default = 50000)')

    parser.add_argument('-r', '--render',
                        type=bool,
                        default=False,
                        help='Whether to render the mountain car using the OpenAI Gym environment (default = False)')

    parser.add_argument('-b', '--bins',
                        type=int,
                        default=70,
                        help='Number of bins used for discretization.')

    parser.add_argument('-s', '--steps',
                        type=int,
                        default=2000,
                        help='Number of time steps each episode.')
    parser.add_argument('--constant',
                        type=bool,
                        default=False,
                        help='The constant printing variable (True/False)')
    parser.add_argument('--debug',
                        type=bool,
                        default=True,
                        help='The Testing mode variable (True/False)')
    parser.add_argument('--agent',
                        default="QLearning",
                        help='Which agent to train (options: QLearning, SARSA, Both) - note that "Both" train the Q learning agent first.')
    # Execute parse_args()
    args = parser.parse_args()

    alpha = args.a
    gamma = args.g
    RENDER = args.render
    max_episodes = [args.episodes]
    bins = [args.bins]
    max_episode_steps = args.steps
    DEBUG = args.debug
    CONSTANT = args.constant
    ag = args.agent


    # env = gym.make("MountainCar-v0")

    gym.envs.register(id='MountainCarMyEasyVersion-v0', entry_point='gym.envs.classic_control:MountainCarEnv',
                      max_episode_steps=max_episode_steps, )  # MountainCar-v0 uses 200
    env = gym.make('MountainCarMyEasyVersion-v0')

    #max_episodes = np.arange(50000, 100000, 10000, dtype=np.int32)
    #bins = np.arange(50, 1000, 50, dtype=np.int32)
    #max_episodes = [10000]
    #bins = [50]
    agent = []
    if ag == "QLearning":
        agent.append(QLearning(env, bins[0], max_episodes[0], epsilon))
    elif ag == "SARSA":
        agent.append(SARSA(env, bins[0], max_episodes[0], epsilon))
    elif ag == "Both":
        agent.append(QLearning(env, bins[0], max_episodes[0], epsilon))
        agent.append(SARSA(env, bins[0], max_episodes[0], epsilon))


    handler = [Agent(env, agent[i]) for i in range(len(agent))]
    policy = [handler[i].train() for i in range(len(handler))]

    env.close()
