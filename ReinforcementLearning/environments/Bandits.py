import numpy as np
import gym


class Bandits(gym.Env):

    def __init__(self, mu=0, sigma=1, number_of_bandits=10):

        self.true_values = np.random.normal(mu, sigma, number_of_bandits)  # number of bandits
        print("bandits initialized with {}".format(self.true_values))

    def get_action():
        pass

    def update():
        pass

    def step(self, action):

        bandit = self.true_values[action]
        reward = np.random.normal(bandit, 0.1)

        return reward

    def reset(self):

        pass
