import random
from collections import defaultdict


class PolicyIteration:
    def __init__(self, actions):
        self.__actions = actions
        self.__eps = 0.5
        self.__alpha = 0.01
        self.__gamma = 0.5
        self.__V = defaultdict(lambda: random.random())

    def get_action(self, state):
        
        # _reward, max_action = max(actions)
        return max_action

    def update(self, state, action, next_state, reward):
        _reward = self.__Q[(state, action)]
        max_next_action_reward = max([self.__Q[(next_state, action)] for action in self.__actions])
        self.__V[(state, action)] = _reward + self.__alpha*(reward + self.__gamma*max_next_action_reward - _reward)



import matplotlib.pyplot as plt
from environments.SimpleGridworld import WorldEnv


world = [
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
]
env = WorldEnv(world)
agent = PolicyIteration(['North', 'South', 'West', 'East'])
for episode in range(30):
    done = False
    state = env.reset()
    print(episode)
    i = 0
    while not done and i<8000:
        i += 1
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, next_state, reward)
        state = next_state
plt.show()

