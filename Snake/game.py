import pygame
import numpy as np
from pygame.locals import *
import helpers
import math

clock = pygame.time.Clock()

# create board class
class Board:
    """
        Class that represents the environment in which the game will be played
    """

    def __init__(self, screen, board):
        self.screen = screen
        self.board = np.zeros([board.shape[0]+1, board.shape[1]+1])

        self.food = []
        self.snake = Snake(screen, board)
        place_food(self)

        # left, right, up, down
        self.actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.direction = [1, 0]


    def draw(self):

        # start with black background
        self.screen.fill(helpers.BLACK)

        # extract the shape
        x, y = self.board.shape

        # we set x-1 and y-1 to refrain from drawing the border
        for i in range(x-1):
            for j in range(y-1):

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

    def step(self, action):
        """
        Method to perform an action given a state
        :param action: the action to perform
        :return: [next_state, reward, done]
        """

        # save old coordinates
        old_coordinates = self.snake.head_coordinates

        # save the old state
        old_state = self.state()

        # change the direction according to the action
        self.direction = self.change_direction(action)

        # move the snake
        food = self.snake.move(self.direction, self)

        # reward system
        reward = self.reward(old_coordinates, food)

        # get next state
        next_state = self.state()

        # if we dont have a next state simply use the old one
        if not next_state:
            next_state = old_state

        # [next_state, reward, done]
        return next_state, reward, self.snake.dead

    def state(self):
        """
        Function to get the state
        A state looks like this:
            [
                Angle of head to fruit,
                left neighbour,
                top neighbour,
                right neighbour
            ]
        """

        # check if the snake is still alive, if not there is no state to return to
        if self.snake.dead:
            return []

        # create new state
        state = []

        # get the angle to the fruit
        angle = self.food_angle()

        # add the angle to the state
        state.append(angle)

        # need to check whether there is a 1 on the top, left or right side of the snake at its current location
        x, y = self.snake.head_coordinates

        # get the direction of left (action 0)
        for a in range(3):

            # get coordinate difference wrt direction
            dir = self.change_direction(a)

            # add coordinates to snake head and check if there is a 1 on the board
            # body collision
            body_col = self.board[x + dir[0], y + dir[1]] == 1

            # out of bounds
            oub = False
            if x + dir[0] < 0 or x + dir[0] > (self.board.shape[0] - 2)\
                    or y + dir[1] < 0 or y + dir[1] > (self.board.shape[1] - 2):
                oub = True

            # add the indicator to the state
            state.append(int(body_col or oub))

        return state

    def reward(self, old_coord, food):
        """
        Method to define the reward for the action given the state
        :return: reward
        """

        # first we check if the snake managed to get food
        if food:
            return 1

        # check for collision
        if collision_detector(self.snake, self):
            return -1

        # set reward for moving closer/further from fruit
        old_dist = self.food_distance(old_coord)
        new_dist = self.food_distance(self.snake.head_coordinates)

        # check if closer or further from fruit
        if old_dist > new_dist:
            return 0.1
        else:
            return -0.2


    def reset(self):
        """
        Method to reset the game and retrieve the first state
        :return: state
        """

        # refresh the environment
        self.board = np.zeros(self.board.shape)
        self.food = []
        self.snake = Snake(self.screen, np.zeros([self.board.shape[0]-1, self.board.shape[1]-1]))
        place_food(self)

        # get state
        return self.state()

    def change_direction(self, action):
        """
        convert action into proper direction
        this code looks quite complicated, but all it does is change the direction
        according to the moves "left", "right" and "forward" that a snake can take
        since forward just leaves the direction to be the same, we don't handle that
        """

        # extract the current direction
        direction = self.direction

        # swap the actions if the snake turns left/right
        if action != 1:

            # check if index is on 0, if so we invert
            if not direction.index(0):
                direction = list(np.array(direction) * -1)

            # change direction
            direction = [direction[1], direction[0]]

        # if the snake turns right, we have the same operations as before (left), but we need to invert it
        if action == 0:
            direction = list(np.array(direction) * -1)

        return direction

    def food_angle(self):
        """
        TODO MIGHT BE ABLE TO SPEED THIS UP
        Method to get the angle of the head to the food
        :return: angle (normalized)
        """

        # get the angle
        angle = 0

        # get coordinates of snake head
        snake_x, snake_y = self.snake.head_coordinates

        # get coordinates of food
        food_x, food_y = self.food

        # get direction to define starting point of angle
        # maybe dictionary to convert direction list to degrees
        rad = math.atan2(snake_y - food_y, snake_x - food_x)

        # convert radians to angle
        angle = math.degrees(rad)

        # keep angle in same domain
        angle = (angle+360) % 360

        # convert direction to degrees
        dir_conv = {(0, -1): 90, (-1, 0): -180, (0, 1): -90, (1, 0): 0}

        # move origin
        angle += dir_conv[tuple(self.direction)]
        angle = (angle + 180) % 360

        # now normalize everything over 180 should be negative
        if angle > 180:
            angle = (angle - 180) * -1

            # normalize
            angle = -1 - (angle/180)

        # normalize
        else:
            angle = angle/180

        return angle

    def food_distance(self, snake_coord):
        """
        Method to calculate the distance between snake and food
        :param snake_coord: the coordinates of the snake head
        :return: distance
        """

        # get coordinates of snake head
        snake_x, snake_y = snake_coord

        # get coordinates of food
        food_x, food_y = self.food

        # get distance
        D = math.sqrt(abs(snake_x-food_x) ** 2 + abs(snake_y-food_y) ** 2)

        return D


class Snake:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, board):
        self.screen = screen
        self.board = board
        self.body = []
        self.length = 3
        self.head_coordinates = [round((board.shape[0]-1)/2), round((board.shape[1]-1)/2)]
        self.dead = False

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

        # collision detection
        self.dead = collision_detector(self, env)

        # check if head hits food
        if self.head_coordinates == env.food:
            self.length += 1
            place_food(env)
            return True

        return False

class Bodypart:
    """
        Class that represents the snake that can be controlled by the user
    """
    def __init__(self, screen, location):
        self.screen = screen
        self.location = location
        self.old_location = location


def place_food(env):
    """
    This function places a new bit of food
    :param env:
    :return:
    """

    x, y = env.board.shape

    # make sure food does not land on border
    x -= 1
    y -= 1

    # randomly generate number in space
    x_coord = np.random.randint(0, x)
    y_coord = np.random.randint(0, y)

    # check if number not already part of snake
    while env.board[x_coord, y_coord]:
        x_coord = np.random.randint(0, x)
        y_coord = np.random.randint(0, y)

    # place food on board
    env.board[x_coord, y_coord] = 2
    env.food = [x_coord, y_coord]

# todo create game rules

def collision_detector(snake, env):
    """ This function checks if the snake collides with itself or the walls"""

    # check if snake head collides into its own body
    for part in snake.body:
        if snake.head_coordinates == part.location:
            return True

    # check if snake leaves game area
    if snake.head_coordinates[0] < 0 or snake.head_coordinates[0] > (env.board.shape[0] - 2):
        return True
    if snake.head_coordinates[1] < 0 or snake.head_coordinates[1] > (env.board.shape[1] - 2):
        return True

    # return false as default
    return False


def main():

    # create game
    pygame.init()
    board = np.zeros([20, 15])
    screen_size = (board.shape[0]*30, board.shape[1]*30)

    # create screen
    DISPLAY = pygame.display.set_mode(screen_size, 0, 32)
    DISPLAY.fill(helpers.BLACK)

    # create snake
    snake = Snake(DISPLAY, board)

    # create board and randomly place food
    env = Board(DISPLAY, board, snake)
    place_food(env)

    # draw environment
    env.draw()

    # set initial direction
    direction = "RIGHT"
    xy_move = [1, 0]

    done = False
    i = 0
    while not done:
        df = clock.tick(8)

        # create background
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

        # check if snake is still alive
        done = snake.dead

        # draw new board state
        env.draw()

        # update display
        pygame.display.update()

#
# main()


# so the center of every point in the array is 15, width is 28