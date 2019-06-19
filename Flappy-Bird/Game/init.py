import pygame, sys, math, helpers, random, time
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
        self.maxSpeed = 10
        self.acc = 1
        self.location = self.x, self.y
        self.speed = vel[0]
        self.angle = vel[1]
        self.playerFlapped = False

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
            self.speed = -15

            # return to normal
            self.playerFlapped = False

        # todo add boundaries
        self.y += self.speed

        self.location = int(self.x), int(self.y)

        # drawing
        pygame.draw.circle(self.screen, self.colour, self.location, self.mass, 0)

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

        self.gap = 200
        self.width = 100

        # generate a valid sample of upper y,
        self.upper_y = random.randint(round(0.1 * helpers.SCREEN_HEIGHT), round(0.7 * helpers.SCREEN_HEIGHT))

        # use gap to detect lower y
        self.lower_y = self.upper_y + self.gap

        self.location = self.x
        self.speed = 3


    def update(self):

        # let x decrease
        self.x -= self.speed

        # use same x to draw two rectangles
        # draw upper
        pygame.draw.rect(self.screen, self.colour, pygame.Rect(self.x, 0, self.width, self.upper_y))

        # draw lower
        pygame.draw.rect(self.screen, self.colour, pygame.Rect(self.x, self.lower_y, self.width, helpers.SCREEN_HEIGHT))


def main():
    # start game
    pygame.init()

    # initialize game settings
    DISPLAY = pygame.display.set_mode(helpers.SCREEN_SIZE,0,32)
    DISPLAY.fill(helpers.WHITE)

    # create user controlled bird
    bird = Bird(DISPLAY, helpers.GREEN, 10, [100,100], [0,0])

    pipes = []
    pipes.append(PipePairs(DISPLAY, helpers.GREEN, helpers.SCREEN_WIDTH))

    # play game until finished
    done = False
    while not done:
        # check all events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # check for user input
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # flap the bird
                    bird.playerFlapped = True

        DISPLAY.fill(helpers.WHITE)
        bird.update()
        for i, pipe in enumerate(pipes):
            pipe.update()

            # for final pipe, check the x coordinate
            if i == len(pipes) - 1:
                # if x coordinate reaches 80% of screen, spawn new pipes
                if pipe.x < 0.6 *helpers.SCREEN_WIDTH:
                    pipes.append(PipePairs(DISPLAY, helpers.GREEN, helpers.SCREEN_WIDTH))

        # delete pipes that are well beyond reach by saving their index and deleting after loop
        if pipes[0].x < 0 - pipes[0].gap + 50:
            del pipes[0]

        pygame.display.update()

main()