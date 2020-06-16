import pygame
import numpy as np
from pygame.locals import *
import helpers
import math
from pygame.math import Vector2
import os

clock = pygame.time.Clock()

# TODO s ################
# - drifting
# - circuit
# - collision
# - state
# - movement

class Car:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, x, y):
        self.screen = screen
        self.acceleration = 0
        self.velocity = 0
        self.length = 60
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = 0
        self.steering = 0

        self.max_acceleration = 5.0
        self.max_steering = 30

    def update(self, dt):
        """
        Function to update car position
        :return:
        """

        # get new speed
        self.velocity += (self.acceleration * dt, 0)

        # steer the car
        self.steer(dt)


    def steer(self, dt):
        """
            Function used to steer the car
        """

        # steer the car
        if self.steering:
            turning_radius = self.length / math.sin(math.radians(self.steering))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        # change position depending on steering
        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += math.degrees(angular_velocity) * dt


class Game:
    """
        Class that represents the snake that can be controlled by the user
    """

    def __init__(self, screen, car, car_image):
        self.screen = screen
        self.car = car
        self.car_image = car_image

    def draw(self):
        self.screen.fill(helpers.BLACK)

        # rotate the car
        rotated = pygame.transform.rotate(self.car_image, self.car.angle)
        rect = rotated.get_rect()
        self.screen.blit(rotated, self.car.position - (rect.width / 2, rect.height / 2))

        # pygame.draw.rect(self.screen, helpers.WHITE,
        #                  pygame.Rect(self.car.position[0],  # x coordinate
        #                              self.car.position[1],  # y coordinate
        #                              self.car.length, 30))  # size

        self.screen.blit(rotated, self.car.position * 30 - (rect.width / 2, rect.height / 2))


def main():

    # create game
    pygame.init()

    screen_size = (1200, 800)

    display = pygame.display.set_mode(screen_size, 0, 32)
    display.fill(helpers.BLACK)

    # create car
    car = Car(display, 50, 400)

    # load car image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "car.png")
    car_image = pygame.image.load(image_path).convert_alpha()
    car_image = pygame.transform.scale(car_image, (60, 30))

    game = Game(display, car, car_image)

    game.draw()

    dt = .01

    done = False
    while not done:

        # check all events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        # get pressed key
        pressed = pygame.key.get_pressed()

        # speed up
        if pressed[pygame.K_UP]:

            if car.velocity.x < 0:
                car.acceleration = 10
            else:
                car.acceleration += 1 * dt

        # make braking faster
        elif pressed[pygame.K_DOWN]:

            if car.velocity.x > 0:
                car.acceleration = -10
            else:
                car.acceleration -= 1 * dt
        else:
            car.acceleration = 0

        car.acceleration = max(-car.max_acceleration, min(car.acceleration, car.max_acceleration))

        #########
        # Steer #
        #########

        # speed up
        if pressed[pygame.K_LEFT]:
            car.steering += 30 * dt

        # make braking faster
        elif pressed[pygame.K_RIGHT]:
            car.steering -= 30 * dt
        else:
            car.steering = 0
        car.steering = max(-car.max_steering, min(car.steering, car.max_steering))

        car.update(dt)

        # draw current state of the game
        game.draw()

        # update display
        pygame.display.update()

main()