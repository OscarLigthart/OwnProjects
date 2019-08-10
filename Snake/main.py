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
        self.food = []


    def draw(self):

        x, y = self.board.shape
        for i in range(x):
            for j in range(y):

                # draw snake part
                if self.board[i, j] == 1:
                    pygame.draw.rect(self.screen, helpers.WHITE,
                                     pygame.Rect(i*helpers.SCREEN_RATIO,
                                                 j*helpers.SCREEN_RATIO,
                                                 helpers.SCREEN_RATIO - 2, helpers.SCREEN_RATIO - 2))

                # draw food
                elif self.board[i, j] == 2:
                    pygame.draw.rect(self.screen, helpers.RED,
                                     pygame.Rect(i * helpers.SCREEN_RATIO,
                                                 j * helpers.SCREEN_RATIO,
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
        self.head_coordinates = [0, 0]

    def move(self, xy_move, env):
        """
        This function moves the snake head and lets the body follow
        :param xy_move:
        :param env:
        :return:
        """
        # move the head in this function. Make sure every bodypart is connected using pointers?
        # use all old_locations for the new locations of the snake
        old_head_coordinate = self.head_coordinates

        # move the head in the direction specified by changes in coordinates
        self.head_coordinates = [sum(pair) for pair in zip(self.head_coordinates, xy_move)]

        new_coordinates = old_head_coordinate

        ######################
        # MOVE ALL BODYPARTS #
        ######################
        for i, part in enumerate(self.body):
            # recursively call bodypart move
            part.location = new_coordinates

            # use the parts old location as coordinates for another bodypart
            new_coordinates = part.old_location

        ###################
        # INCREASE LENGTH #
        ###################
        # add a bodypart if the snake does not have the set length
        if len(self.body) < self.length:
            # add new body part if empty
            if len(self.body) == 0:
                # add body part on old location head
                self.body.append(Bodypart(self.screen, old_head_coordinate))

            # add one
            else:
                # add body part on location of snake tail (do not move this part)
                self.body.append(Bodypart(self.screen, self.body[-1].old_location))

        # update old locations of bodyparts
        for part in self.body:
            # change the parts old location
            part.old_location = part.location

        ########################
        # PLACE SNAKE ON BOARD #
        ########################

        # remember where the food was
        food_ind = np.where(np.array(env.board) == 2)
        env.food = [food_ind[0][0], food_ind[1][0]]

        # first empty the old board
        env.board = np.zeros(env.board.shape)

        # put food on board
        env.board[food_ind] = 2

        # put head on board
        env.board[tuple(self.head_coordinates)] = 1

        # put body on board
        for part in self.body:
            env.board[tuple(part.location)] = 1



class Bodypart:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, location):
        self.screen = screen
        self.location = location
        self.old_location = location

def eat_food(snake, env):
    """
    This function checks whether the head of the snake eats the food
    :param snake:
    :param env:
    :return:
    """
    # check if head location matches food location
    if snake.head_coordinates == env.food:
        snake.length += 1
        place_food(env)

# todo create food class
def place_food(env):
    """
    This function places a new bit of food
    :param env:
    :return:
    """

    x, y = env.board.shape

    # randomly generate number in space
    x_coord = np.random.randint(0, x)
    y_coord = np.random.randint(0, y)

    # check if number not already part of snake
    while env.board[x_coord, y_coord]:
        x_coord = np.random.randint(0, x)
        y_coord = np.random.randint(0, y)

    # place food on board
    env.board[x_coord, y_coord] = 2

# todo create game rules

def collision_detector(snake, env):
    """ This function checks if the snake collides with itself or the walls"""

    # check if snake head collides into its own body
    for part in snake.body:
        if snake.head_coordinates == part.location:
            quit()

    # check if snake leaves game area
    if snake.head_coordinates[0] < 0 or snake.head_coordinates[0] > 29:
        quit()
    if snake.head_coordinates[1] < 0 or snake.head_coordinates[1] > 24:
        quit()




def main():

    pygame.init()
    board = np.zeros([20, 15])
    screen_size = (board.shape[0]*30, board.shape[1]*30)

    # create screen
    DISPLAY = pygame.display.set_mode(screen_size, 0, 32)
    DISPLAY.fill(helpers.BLACK)

    # create board and randomly place food
    env = Board(DISPLAY, board)
    place_food(env)

    # create snake
    snake = Snake(DISPLAY, board)

    # draw environment
    env.draw()

    # set initial direction
    direction = "RIGHT"
    xy_move = [1, 0]

    done = False
    i = 0
    while not done:

        df = clock.tick(8)
        DISPLAY.fill(helpers.BLACK)
        # check all events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

            ########################
            # check for user input #
            ########################

            if event.type == pygame.KEYDOWN:
                # check if direction is not opposing for all moves, then set it
                if event.key == pygame.K_LEFT and direction != "RIGHT":
                    direction = "LEFT"

                    # save x and y changes in coordinates
                    xy_move = [-1, 0]

                elif event.key == pygame.K_RIGHT and direction != "LEFT":
                    direction = "RIGHT"

                    # save x and y changes in coordinates
                    xy_move = [1, 0]

                elif event.key == pygame.K_UP and direction != "DOWN":
                    direction = "UP"

                    # save x and y changes in coordinates
                    xy_move = [0, -1]

                elif event.key == pygame.K_DOWN and direction != "UP":
                    direction = "DOWN"

                    # save x and y changes in coordinates
                    xy_move = [0, 1]

        # update snake
        snake.move(xy_move, env)

        # food detection
        eat_food(snake, env)

        # collision detection
        collision_detector(snake, env)

        # draw new board state
        env.draw()

        pygame.display.update()


main()


# so the center of every point in the array is 15, width is 28