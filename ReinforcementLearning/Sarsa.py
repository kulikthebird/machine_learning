import random
from collections import defaultdict


class Sarsa:
    def __init__(self, actions, eps=0.99, alpha=0.1, gamma=0.9):
        self.__actions = actions
        self.__eps = eps
        self.__alpha = alpha
        self.__gamma = gamma
        self.__Q = defaultdict(lambda: random.normalvariate(0., 1.))

    def get_action(self, state):
        self.__eps = self.__eps*0.999
        if random.random() < self.__eps:
            return random.choice(self.__actions)
        actions = [(self.__Q[(state, action)], action) for action in self.__actions]
        _reward, max_action = max(actions)
        return max_action

    def update(self, state, action, next_state, next_action, reward):
        _prev_reward = self.__Q[(state, action)]
        self.__Q[(state, action)] = _prev_reward + self.__alpha*(reward + self.__gamma*self.__Q[(next_state, next_action)] - _prev_reward)

    def learn_episode(self, env, max_steps_per_episode, should_render):
        done = False
        steps = 0
        rewards = 0
        state = env.reset()
        action = self.get_action(state)
        while not done and steps < max_steps_per_episode:
            steps += 1
            next_state, reward, done, _info = env.step(action)
            rewards += reward
            next_action = self.get_action(next_state)
            self.update(state, action, next_state, next_action, reward)
            state, action = next_state, next_action
            if should_render:
                env.render(0.02)
        return steps, rewards


def run_sarsa(env, actions, episodes, max_steps_per_episode):
    agent = Sarsa(actions)
    rewards = []
    steps_taken = []
    for episode in range(episodes):
        print(episode)
        steps, reward = agent.learn_episode(env, max_steps_per_episode=max_steps_per_episode, should_render=(episode == episodes-1))
        rewards += [reward]
        steps_taken += [steps]
    return steps_taken, rewards



import matplotlib.pyplot as plt
from environments.GridWorld4 import Environment


env = Environment(size=(10, 7),
                   obstacles=[],
                   agent_pos=(0, 3),
                   reward_pos=(7, 3),
                   horizon=0
                   )

steps, _ = run_sarsa(env=env, actions=env.directions, episodes=1000, max_steps_per_episode=500)
plt.plot(steps)
plt.show()
