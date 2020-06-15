import pygame
import numpy as np
from pygame.locals import *
from game import Board, Snake, Bodypart, collision_detector, place_food
from model import *
import helpers

clock = pygame.time.Clock()


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, df):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):

        t = 0
        state = env.reset()

        # Take actions until end of episode
        for t in range(1000):

            df = clock.tick(5)

            # create background
            epsilon = get_epsilon(global_steps)

            action = select_action(model, state, epsilon)

            next_state, reward, done = env.step(action)

            memory.push((state, action, reward, next_state, done))

            # only sample if there is enough memory
            if len(memory) > batch_size:
                loss = train(model, memory, optimizer, batch_size, discount_factor)

            state = next_state
            global_steps += 1

            # draw new board state
            env.draw()

            # update display
            pygame.display.update()

            if done:
                break

        episode_durations.append(t)

    return episode_durations

def main():

    # create game
    pygame.init()
    board = np.zeros([20, 15])
    screen_size = (board.shape[0] * 30, board.shape[1] * 30)

    # create screen
    DISPLAY = pygame.display.set_mode(screen_size, 0, 32)
    DISPLAY.fill(helpers.BLACK)

    # create board and randomly place food
    env = Board(DISPLAY, board)

    # draw environment
    env.draw()

    # Let's run it!
    num_episodes = 100
    batch_size = 4 #32
    discount_factor = 0.8
    learn_rate = 1e-3
    memory = ReplayMemory(1000)
    num_hidden = 128
    df = 8

    model = QNetwork(num_hidden)

    episode_durations = run_episodes(train, model, memory, env,
                                     num_episodes, batch_size,
                                     discount_factor, learn_rate, df)



main()