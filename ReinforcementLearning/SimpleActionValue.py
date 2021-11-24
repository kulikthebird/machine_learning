import random


class SimpleActionValue:
    def __init__(self, length, eps):
        self.__bandits = [0]*length
        self.__eps = eps
        self.__alpha=0.01

    def get_action(self):
        self.__eps = self.__eps*0.99
        if random.random() < self.__eps:
            return random.randint(0, len(self.__bandits)-1)
        return self.__bandits.index(max(self.__bandits))

    def update(self, action, r):
        self.__bandits[action] += self.__alpha*(r-self.__bandits[action])


def run(env, start_epsilon):
    agent = SimpleActionValue(len(env.true_values), start_epsilon)
    i, rewards = 0, []
    while i < 10000:
        i += 1
        action = agent.get_action()
        r = env.step(action)
        agent.update(action, r)
        rewards.append(r)
    return rewards



from environments.Bandits import Bandits
import matplotlib.pyplot as plt

for epsilon, color in zip((0.01, 0.1, 1), ('r', 'g', 'b')):
    result = run(Bandits(), epsilon)
    plt.plot(result, color)
plt.show()
plt.legend()
