# base class for 6.6 - cliff walking
# windy gridworld 6.5
# 7.4
# 8.2
# 8.4
import pygame
import gym
import time

from .Cell import Cell


class Environment(gym.Env):
    MOVE_REWARD = 0
    INVALID_MOVE = -2
    GOAL_REWARD = 10
    WHITE = (200, 200, 200)
    GREEN = (0, 200, 0)
    RED = (200, 0, 0)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)

    def __init__(self, size, obstacles, agent_pos, reward_pos, horizon):
        self.steps = 0
        self.horizon = horizon
        self.obstacles = obstacles
        self.reset_agent_pos = agent_pos
        self.reward_cell = reward_pos
        self.cells = {}
        self.directions = ['S', 'N', 'E', 'W']
        size_x, size_y = size
        for i in range(size_x):
            for j in range(size_y):
                self.cells[(i, j)] = Cell(i, j)
        self.agent_cell = self.cells[agent_pos]
        self.reward_cell = self.cells[reward_pos]

        # Add neighbours for each cell
        increments = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for inc, direction in zip(increments, self.directions):
            for coords, cell in self.cells.items():
                neighbour_coords = coords[0] + inc[0], coords[1] + inc[1]

                try:
                    other_cell = self.cells[neighbour_coords[0], neighbour_coords[1]]
                except KeyError:  # edge cell, no neighbour in this direction
                    other_cell = None

                if neighbour_coords not in self.obstacles:
                    cell.add_neighbour(other_cell, direction)
                else:
                    cell.add_neighbour(None, direction)

        self.done = False

    def step(self, direction):
        self.steps += 1
        reward = self.MOVE_REWARD

        if direction == 'N':
            if self.agent_cell.north_neighbour is not None:
                self.agent_cell = self.agent_cell.north_neighbour

        elif direction == 'S':
            if self.agent_cell.south_neighbour is not None:
                self.agent_cell = self.agent_cell.south_neighbour

        elif direction == 'E':
            if self.agent_cell.east_neighbour is not None:
                self.agent_cell = self.agent_cell.east_neighbour

        elif direction == 'W':
            if self.agent_cell.west_neighbour is not None:
                self.agent_cell = self.agent_cell.west_neighbour

        if self.agent_cell == self.reward_cell:
            reward += self.GOAL_REWARD
            self.done = True

        if self.steps == self.horizon:  # if horizon is 0, this is never satisfied
            self.done = True

        return (self.agent_cell.x, self.agent_cell.y), reward, self.done, {}

    def reset(self):
        agent_pos = self.reset_agent_pos
        self.agent_cell = self.cells[agent_pos]
        self.done = False
        self.steps = 0
        return agent_pos

    def render(self, pause):
        Cell.SCREEN.fill(self.BLACK)

        for key, cell in self.cells.items():

            if cell == self.reward_cell:
                cell.draw(self.RED, 0)

            elif cell == self.agent_cell:
                cell.draw(self.GREEN, 0)

            elif key in self.obstacles:
                cell.draw(self.GRAY, 0)

            else:
                cell.draw(self.WHITE, 1)

        pygame.display.update()
        time.sleep(pause)
