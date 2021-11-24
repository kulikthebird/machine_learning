import random
from collections import defaultdict


class DynaQ:
    def __init__(self, actions, planning_steps, eps=0.99, alpha=0.1, gamma=0.9):
        self.__actions = actions
        self.__eps = eps
        self.__alpha = alpha
        self.__gamma = gamma
        self.__Q = defaultdict(lambda: random.random())
        self.__planning_steps = planning_steps
        self.__model = defaultdict(lambda: ((0, 0), random.random()))

    def get_action(self, state):
        self.__eps = self.__eps*0.99
        if random.random() < self.__eps:
            return random.choice(self.__actions)
        actions = [(self.__Q[(state, action)], action) for action in self.__actions]
        _reward, max_action = max(actions)
        return max_action

    def __add_to_model(self, state, action, next_state_, reward):
        self.__model[(state, action)] = (next_state_, reward)

    def update(self, state, action, next_state, reward):
        self.__add_to_model(state, action, next_state, reward)
        for _ in range(self.__planning_steps):
            state, action = random.choice(list(self.__model.keys()))
            next_state, reward = self.__model[(state, action)]
            self.updateQ(state, action, next_state, reward)
        else:
            self.updateQ(state, action, next_state, reward)

    def updateQ(self, state, action, next_state, reward):
        _reward = self.__Q[(state, action)]
        max_next_action_reward = max([self.__Q[(next_state, action)] for action in self.__actions])
        self.__Q[(state, action)] = _reward + self.__alpha*(reward + self.__gamma*max_next_action_reward - _reward)



def run_dyna_q(env, actions, episodes, max_steps_per_episode, planning_steps):
    agent = DynaQ(actions, planning_steps)
    rewards = 0
    steps_taken = []
    for episode in range(episodes):
        done = False
        steps = 0
        state = env.reset()
        print(episode)
        while not done and steps < max_steps_per_episode:
            steps += 1
            action = agent.get_action(state)
            next_state, reward, done, _info = env.step(action)
            agent.update(state, action, next_state, reward)
            rewards += reward
            state = next_state
            if episode == episodes-1:
                env.render(0.02)
        steps_taken += [steps]
    return steps_taken


from environments.GridWorld import Environment
import matplotlib.pyplot as plt


env = Environment(size=(9, 6),
                  obstacles=[(2, 1), (2, 2), (2, 3),
                             (5, 4),
                             (7, 0), (7, 1), (7, 2)],
                  agent_pos=(0, 2),
                  reward_pos=(8, 0),
                  horizon=2000
                  )
for color, plan_steps in zip(('g', 'r', 'b'), (0, 10, 25)):
    result = run_dyna_q(env, env.directions, episodes=50, max_steps_per_episode=8000, planning_steps=plan_steps)
    plt.plot(result, color)
plt.show()
