from collections import defaultdict
import random


class MonteCarloFirst:
    def __init__(self, actions, eps=0.99, alpha=0.1, gamma=0.9):
        self.__actions = actions
        self.__eps = eps
        self.__alpha = alpha
        self.__gamma = gamma
        self.__V = defaultdict(lambda: random.random())
        # self.__model = defaultdict(lambda: ((0, 0), random.random()))

    def get_action(self, state):
        self.__eps = self.__eps*0.99
        if random.random() < self.__eps:
            return random.choice(self.__actions)
        actions = [(self.__V[(state, action)], action) for action in self.__actions]
        _reward, max_action = max(actions)
        return max_action

    def update(self, state, G):
        if state not in self.V:
            self.__V[state] = 0.5
        self.__V[state] = self.__V[state] + 0.09 * (G - self.V[state])

