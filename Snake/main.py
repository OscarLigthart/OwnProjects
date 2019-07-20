import pygame
import numpy as np
from pygame.locals import *


import helpers

clock = pygame.time.Clock()

# create board class
class Board:
    """
        Class that represents the environment in which the game will be played
    """

    def __init__(self, screen, board):
        self.screen = screen
        self.board = board


    def draw(self):

        x, y = self.board.shape
        print(x, y)
        for i in range(x):
            for j in range(y):
                if self.board[i, j]:
                    pygame.draw.rect(self.screen, helpers.WHITE,
                                     pygame.Rect(i*helpers.SCREEN_RATIO,
                                                 j*helpers.SCREEN_RATIO,
                                                 helpers.SCREEN_RATIO - 2, helpers.SCREEN_RATIO - 2))


# create snake class
class Snake:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
        self.body = []
        self.length = 3

    def move(self):
        # move the head in this function. Make sure every bodypart is connected using pointers?
        # use all old_locations for the new locations of the snake
        dt = clock.tick(1000)

        # move the head in the direction the user wants to move, let's say right this time


        while len(self.body) < self.length:
            # todo keep adding body parts while moving the head

class bodypart:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, location):
        self.screen = screen
        self.old_location = location

    def move(self):

# create food class


# create game rules

# start with a 25*30 or something

def main():

    pygame.init()
    board = np.zeros([25, 30])
    board[5,6] = 1
    board[5,7] = 1

    DISPLAY = pygame.display.set_mode(helpers.SCREEN_SIZE, 0, 32)
    DISPLAY.fill(helpers.BLACK)

    env = Board(DISPLAY, board)

    env.draw()

    done = False
    i = 0
    while not done:
        pygame.display.update()
        i += 1
        if i > 100:
            done = True

main()

print(900/30)
print(750/25)

# so the center of every point in the array is 15, width is 28