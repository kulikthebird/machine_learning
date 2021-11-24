# coding=utf-8
# compatible with OpenAI gym
import copy
import time

import gym

# 1. Use numpy instead of lists. Measure the difference in speed
# 2. Increase the size of the environment. Observe learning speed. Compare with numpy
# 3. Change the state representation such that one state is identified by cartesian coordinates (x, y)


class WorldEnv(gym.Env):

    def __init__(self, World):
        self.world_initial = [copy.deepcopy(x[:]) for x in World]
        self.world = copy.deepcopy(World)
        self.agent_pos = None
        self.rewards = copy.deepcopy([[element - 1 for element in row] for row in World])

    def step(self, action):

        if (action == 'North' and self.agent_pos[0] == 0) or \
                (action == 'South' and self.agent_pos[0] == len(self.world)-1) or \
                (action == 'West' and self.agent_pos[1] == 0) or \
                (action == 'East' and self.agent_pos[1] == len(self.world[0])-1):
            #print('Invalid move {} from pos {}'.format(action, self.agent_pos))
            pass
        else:
            old_pos = self.agent_pos.copy()
            if action == 'North':
                self.agent_pos[0] -= 1
            elif action == 'South':
                self.agent_pos[0] += 1
            elif action == 'East':
                self.agent_pos[1] += 1
            elif action == 'West':
                self.agent_pos[1] -= 1
            # print("Action = {}, agent_pos = {}".format(action, self.agent_pos))

            self.world[self.agent_pos[0]][self.agent_pos[1]] = -1
            self.world[old_pos[0]][old_pos[1]] = 0

        reward = self.rewards[self.agent_pos[0]][self.agent_pos[1]]
        return self.world, reward, True if reward == 0 else False, {}

    def reset(self):
        # print('reset!')
        self.world = copy.deepcopy(self.world_initial)
        self.world[1][2] = -1
        self.agent_pos = [1, 2]

        return self.world
