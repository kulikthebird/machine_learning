# 6.5
from .GridWorld import Environment


class Environment4(Environment):
    def __init__(self, size, obstacles, agent_pos, reward_pos, horizon):
        super(Environment4, self).__init__(size, obstacles, agent_pos, reward_pos, horizon)

    def step(self, direction):
        state, reward, self.done, info = super().step(direction)

        try:
            if self.agent_cell.x in (3, 4, 5, 8):
                self.agent_cell = self.cells[self.agent_cell.x, self.agent_cell.y-1]
            elif self.agent_cell.x in (6, 7):
                self.agent_cell = self.cells[self.agent_cell.x, self.agent_cell.y-2]
        except KeyError:  # agent goes out of bounds
            self.agent_cell = self.cells[self.agent_cell.x, 0]

        if self.agent_cell == self.reward_cell:
            reward += self.GOAL_REWARD
            self.done = True

        return (self.agent_cell.x, self.agent_cell.y), reward, self.done, info
