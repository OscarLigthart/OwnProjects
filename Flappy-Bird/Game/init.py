import pygame, sys, math, helpers, random, time
import numpy as np
from pygame.locals import *

# GLOBALS


clock = pygame.time.Clock()
lines = []

class Bird:
    """
    Class that represents the bird that can be controlled by the player
    """
    def __init__(self,screen,colour, mass, loc, vel):
        self.screen = screen
        self.colour = colour

        self.mass = mass

        self.x = loc[0]
        self.y = loc[1]
        self.maxSpeed = 5
        self.acc = 0.5
        self.location = self.x, self.y
        self.speed = vel[0]
        self.angle = vel[1]
        self.playerFlapped = False
        self.rotation = 0
        self.rotSpeed = 1.5

    def update(self):
        """
        This function is used to update the bird's position, based on input and gravity
        """
        dt = clock.tick(60)

        # check if speed is not exceeding threshold and if no user input is present
        if self.speed < self.maxSpeed and not self.playerFlapped:
            # increase speed by set acceleration
            self.speed += self.acc

        # check if user input is present
        if self.playerFlapped:
            # change speed of bird to represent a flap
            self.speed = -8

            # change rotation of bird
            self.rotation = 45

            # return to normal
            self.playerFlapped = False

        # adjust rotation
        if self.rotation > -90:
            self.rotation -= self.rotSpeed

        # todo add boundaries
        self.y += self.speed

        self.location = int(self.x), int(self.y)

        # drawing
        #pygame.draw.circle(self.screen, self.colour, self.location, self.mass, 0)

class PipePairs:
    """
    This class represents the pairs of pipes which the bird has to dodge
    """
    def __init__(self, screen, colour, x):
        self.screen = screen
        self.colour = colour

        self.x = helpers.SCREEN_WIDTH
        self.upper_y = 0
        self.lower_y = 0

        self.gap = 120
        self.width = 100

        # generate a valid sample of upper y,
        self.upper_y = random.randint(round(0.1 * helpers.SCREEN_HEIGHT), round(0.5 * helpers.SCREEN_HEIGHT))

        # use gap to detect lower y
        self.lower_y = self.upper_y + self.gap

        self.location = self.x
        self.speed = 3


    def update(self):

        # let x decrease
        self.x -= self.speed


def collisionDetector(bird, birdHeight, birdWidth, pipes, pipeHeight, pipeWidth):
    """
    This function checks whether the bird collided with the base or a pipe
    :param bird: the player
    :param pipes: the obstacles
    :return: true or false, depending on collisions
    """

    # first check whether the bird collides with the ground
    if bird.y > 460:
        return [True, 0]
    else:
        # get position of bird
        birdPos = bird.y

        # get position of both pipes
        for pipe in pipes:
            # skip if difference in distance is too high
            if abs(pipe.x - bird.x) > 100:
                continue

            upperPos = pipe.upper_y
            lowerPos = pipe.lower_y

            # todo: create masks instead, because the bird is treated as a square in this collision detector

            # get image alpha

            # use boolean to see which pixel is actually drawn

            # use that mask to detect for collision

            # DO THIS ONCE AT THE START, then use the mask

            # detect for collision
            # check if x is too close
            if bird.x + birdWidth > pipe.x and \
                    bird.x < pipe.x + pipeWidth:

                # check if y exceeds threshold
                if bird.y < upperPos:
                    return [True, 1]
                elif bird.y + birdHeight > lowerPos:
                    return [True, 1]

    return [False, 0]

def main():
    # start game
    pygame.init()

    # initialize game settings
    DISPLAY = pygame.display.set_mode(helpers.SCREEN_SIZE, 0, 32)
    DISPLAY.fill(helpers.WHITE)

    # create user controlled bird
    bird = Bird(DISPLAY, helpers.GREEN, 10, [100,100], [0,0])

    pipes = []
    pipes.append(PipePairs(DISPLAY, helpers.GREEN, helpers.SCREEN_WIDTH))

    ###############
    # load images #
    ###############
    background = pygame.image.load("sprites/background.png").convert()
    flappy = pygame.image.load("sprites/bird.png").convert()
    lower_pipe = pygame.image.load("sprites/pipe.png").convert_alpha()
    upper_pipe = pygame.transform.flip(
                    pygame.image.load("sprites/pipe.png").convert_alpha(), False, True)
    base = pygame.image.load("sprites/base.png").convert()

    pygame.draw.line(DISPLAY, helpers.RED, (0, 480), (400, 480), 2)

    birdWidth = flappy.get_width()
    birdHeight = flappy.get_height()

    pipeWidth = lower_pipe.get_width()
    pipeHeight = lower_pipe.get_height()


    # play game until finished
    done = False
    pipeCrash = False
    while not done:
        # check all events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # check for user input
            if event.type == pygame.KEYDOWN and not pipeCrash:
                if event.key == pygame.K_SPACE:
                    # flap the bird
                    bird.playerFlapped = True

        # HEURISTIC
        # closest pipe
        # todo: remove, the heuristic below
        # for pipe in pipes:
        #     if abs(pipe.x - bird.x) > 120:
        #         continue
        #     if bird.y + birdHeight > pipe.lower_y - 5 and not pipeCrash:
        #         bird.playerFlapped = True
        #     else:
        #         bird.playerFlapped = False

        # load background
        DISPLAY.fill(helpers.WHITE)
        DISPLAY.blit(background, (0, 0))

        # 480 FOR COLLISION DETECTION
        #pygame.draw.line(DISPLAY, helpers.RED, (0, 480), (400, 480), 2)

        # update bird position
        bird.update()

        # apply bird rotation
        flappybird = pygame.transform.rotate(flappy, bird.rotation)

        # update all pipe positions
        for i, pipe in enumerate(pipes):
            pipe.update()

            # insert LOWER pipe with given x and y
            DISPLAY.blit(lower_pipe, (pipe.x, pipe.lower_y))

            # insert UPPER pipe with given x and y
            DISPLAY.blit(upper_pipe, (pipe.x, -upper_pipe.get_height() + pipe.upper_y))

            # for final pipe, check the x coordinate
            if i == len(pipes) - 1:
                # if x coordinate reaches 50% of screen, spawn new pipes
                if pipe.x < 0.5 * helpers.SCREEN_WIDTH:
                    pipes.append(PipePairs(DISPLAY, helpers.GREEN, helpers.SCREEN_WIDTH))

        # delete pipes that are well beyond reach by saving their index and deleting after loop
        if pipes[0].x < 0 - pipes[0].gap + 50:
            del pipes[0]

        # insert bird into frame
        DISPLAY.blit(flappybird, (bird.x, bird.y))

        # show base
        DISPLAY.blit(base, (0, 480))

        # check for collision on nearest pipe
        collide, pipeCrash = collisionDetector(bird, birdHeight, birdWidth, pipes, pipeHeight, pipeWidth)

        # stop the game if the user collided
        if collide:
            if pipeCrash:
                # stop all pipes
                for pipe in pipes:
                    pipe.speed = 0

                # increase bird speed
                bird.acc = 1.5
                bird.maxSpeed = 12
                bird.rotSpeed = 3
            else:
                break

        pygame.display.update()

    # check whether the user collided with a pipe, a death animation follows if so




main()