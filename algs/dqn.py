####################################################
# Description: Deep Q Network
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-17
####################################################
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.memories import ReplayBuffer

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        fc3_units = 64
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.bn1(F.leaky_relu(self.fc1(state), negative_slope=0.1))
        x = self.bn2(F.leaky_relu(self.fc2(x), negative_slope=0.1))
        x = self.bn3(F.leaky_relu(self.fc3(x), negative_slope=0.1))
        x = self.fc4(x)
        return x


class Agent_DQN(object):

    def __init__(self, env, args):
        '''

        :param n_observations: (int) dimension of each state
        :param n_actions: (int) dimension of each action
        :param buffer_size:
        :param bach_size:
        :param seed:
        :param TAU: (float) soft update of q target
        :param UPDATE_EVERY:
        :param GAMMA: (float) discount factor
        :param LR:
        :param gpu:
        '''
        self.env = env
        if not args.cloose_loop:
            self.state_size = env.state_size
        else:
            # TODO: To determined by the transformer
            self.state_size = args.decoder_size

        if not args.spt:
            if args.cont:
                self.action_size = env.action_space.shape[0]
            else:
                self.action_size = env.action_space.n
        else:
            self.action_size = env.action_space.shape[0]
        # self.action_size = env.action_size
        self.batch_size = args.batch_size
        self.LR = args.LR
        self.TAU = args.TAU
        self.GAMMA = args.GAMMA
        self.UPDATE_EVERY = args.UPDATE_EVERY

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

        self.qnet_target = QNetwork(self.state_size, self.action_size, args.seed).to(self.device)
        # Local Qnet
        self.model = QNetwork(self.state_size, self.action_size, args.seed).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.LR)

        # Replay memory
        self.memory = ReplayBuffer(self.state_size, self.action_size, args.buffer_size, self.batch_size, self.device, args.seed)
        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        '''

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        # save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn every UPDATE_EVERY steps
            if self.memory.__len__() > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        '''
        Returns actions for given state as per current policy

        :param state:
        :param eps: (float) epsilon, for epsilon-greedy action selection
        :return:
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from the target model
        Q_target_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.GAMMA * Q_target_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.model(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------------ update the target network ------------------------------ #
        self.soft_update(self.model, self.qnet_target)

    def soft_update(self, local_model, target_model):
        '''
        Soft update model parameters
        theta_target = tau * theta_local + (1 - tau) * theta_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :return:
        '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU * local_param.data + (1 - self.TAU) * target_param.data)