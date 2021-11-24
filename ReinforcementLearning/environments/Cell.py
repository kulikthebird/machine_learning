import pygame


class Cell:
    blockSize = 40
    WINDOW_HEIGHT = 500
    WINDOW_WIDTH = 500
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    WHITE = (200, 200, 200)
    BLACK = (0, 0, 0)
    LINE_WIDTH = 2
    pygame.init()
    SCREEN.fill(BLACK)

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.north_neighbour, self.south_neighbour, self.east_neighbour, self.west_neighbour = None, None, None, None
        self.x_render = x * self.blockSize
        self.y_render = y * self.blockSize
        self.offset = (10, 100)

    def draw(self, color, size):
        rect = pygame.Rect(self.offset[0] + self.x_render, self.offset[1] + self.y_render, self.blockSize, self.blockSize)
        pygame.draw.rect(self.SCREEN, color, rect, size)

    def add_neighbour(self, other_cell, direction):
        if direction == 'N':
            self.north_neighbour = other_cell
        elif direction == 'S':
            self.south_neighbour = other_cell
        elif direction == 'E':
            self.east_neighbour = other_cell
        elif direction == 'W':
            self.west_neighbour = other_cell
