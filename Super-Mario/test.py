from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import random
from helpers import *
import matplotlib.pyplot as plt
from torch.autograd import Variable

# sources
# https://github.com/Kautenja/gym-super-mario-bros
# https://vmayoral.github.io/robots,/ai,/deep/learning,/rl,/reinforcement/learning/2016/08/07/deep-convolutional-q-learning/

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

################### Model #######################

class CNN(nn.Module):
    def __init__(self, n_channels, n_actions):
        super(CNN, self).__init__()

        # convolutions
        self.conv1 = nn.Conv2d(n_channels, 6, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2, 1)

        self.conv2 = nn.Conv2d(6, 16, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2, 1)

        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2, 1)

        self.avgpool = nn.AvgPool2d(1, 1)

        # fully connecteds
        self.fc1 = nn.Linear(32 * 9 * 9, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, n_actions)


    def forward(self, x):
        # convolutions
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.avgpool(x)

        #print(x.shape)
        # fully connecteds
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def run_episodes(env, memory, model, optimizer, batch_size, num_episodes, discount_factor):

    #print(env.action_space.sample())
    # hyperparameters
    epsilon = 0.1

    episode_duration = []
    episode_rewards = []

    # loop for a set amount of actions
    for i in range(num_episodes):
        print("Starting episode nr " + str(i))
        state = env.reset()
        done = False
        # downsample the image
        #state = scipy.misc.imresize(state, 50, 'nearest')
        state = scipy.misc.imresize(state, 25, 'nearest')

        # convert state to tensor and preprocess it for pytorch network
        state = torch.FloatTensor(torch.from_numpy(state).float())
        state = state.permute(2, 0, 1)
        state = state.view(1, state.shape[0], state.shape[1], state.shape[2])

        for step in range(2500):
            if done:
                break

            # run state through model to get predictions
            out = model(state)
            _, action = out.max(1)

            # convert action to compatible action (tensor to numpy)
            action = action.data.numpy()[0]

            # choose best action with probability 1-e (epsilon-greedy)
            # predict the action given the model
            if np.random.random_sample() > epsilon:
                next_state, reward, done, info = env.step(action)
            else:
                next_state, reward, done, info = env.step(env.action_space.sample())  # enter integer between 0 and 11

            # convert next state (by downsampling)
            # downsample the image
            next_state = scipy.misc.imresize(next_state, 25, 'nearest')

            # convert state to tensor and preprocess it for pytorch network
            next_state = torch.FloatTensor(torch.from_numpy(next_state).float())
            next_state = next_state.permute(2, 0, 1)
            next_state = next_state.view(1, next_state.shape[0], next_state.shape[1], next_state.shape[2])

            # experience replay
            # push action into memory
            memory.push((state, action, reward, next_state, done))

            if len(memory) > batch_size:
                loss = train(model, memory, optimizer, batch_size, discount_factor)
                if step % 100 == 0:
                    print(loss)

            # loop through epochs
            state = next_state

            # perform action
            #if i % 4 == 0 and i != 0:
                #env.render()
            #env.render()

        # save model every 25 episodes
        torch.save(model.state_dict(), 'models/mariomodel'+str(i))

    env.close()

    return episode_rewards




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', default='SuperMarioBros-v3', type=str,
                        help='max number of epochs')
    parser.add_argument('-m', default='human', type=str,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = gym_super_mario_bros.make('SuperMarioBrosNoFrameskip-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

    memory = ReplayMemory(10000)
    batch_size = 64
    num_episodes = 100
    discount_factor = 0.90

    # initialize model
    model = CNN(3, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    run_episodes(env, memory, model, optimizer, batch_size, num_episodes, discount_factor)

    # todo FREEZE TARGET NETWORK
    # todo toy with parameters
    # todo toy with network structure
