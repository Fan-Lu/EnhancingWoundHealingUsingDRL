####################################################
# Description: Advantage Actor Critic
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-24
####################################################

# In my experience, A2C works better than A3C and ACKTR is better than both of them.
# Moreover, PPO is a great algorithm for continuous control.
# Thus, I recommend to try A2C/PPO/ACKTR first and use A3C only if you need it specifically for some reasons.

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.memories import TrajectoryBuffer


class A2C_Net(nn.Module):

    def __init__(self, state_size, action_size):
        super(A2C_Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.valueFunc = nn.Linear(128, 1)
        self.policyFunc = nn.Linear(128, self.action_size)

    def forward(self, inputs):

        x = self.bn1(F.elu(self.fc1(inputs)))
        x = self.bn2(F.elu(self.fc2(x)))
        x = self.bn3(F.elu(self.fc3(x)))

        policy = F.softmax(self.policyFunc(x), dim=1)
        value = self.valueFunc(x)

        policy = torch.clamp(policy, 1e-8, 1-1e-8)

        return policy, value


class Agent_A2C(object):

    def __init__(self, env, args):
        '''
        Advantage Actor Critic
        '''
        super(Agent_A2C, self).__init__()
        self.env = env
        if not args.cloose_loop:
            self.state_size = env.state_size
        else:
            # TODO: To determined by the transformer
            self.state_size = args.decoder_size
        # self.state_size = env.state_size
        if not args.spt:
            if args.cont:
                self.action_size = env.action_space.shape[0]
            else:
                self.action_size = env.action_space.n
        else:
            self.action_size = env.action_space.shape[0]
        self.action_dim = 1
        self.batch_size = args.batch_size
        self.LR = args.LR
        self.TAU = args.TAU
        self.GAMMA = args.GAMMA

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

        torch.manual_seed(args.seed)
        self.model = A2C_Net(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.LR)

        # Replay memory
        self.memory = TrajectoryBuffer(self.state_size, self.action_dim, args.buffer_size, self.device)

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

    def act(self, state, eps=1.):
        '''
        Returns actions for given state as per current policy

        :param state:
        :param eps: (float) epsilon, for epsilon-greedy action selection
        :return:
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(state)
        self.model.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # action = np.take(self.env.action_space, np.argmax(policy.cpu().data.numpy()))
            # action = self.env.action_space[np.argmax(policy.cpu().data.numpy())]
            action = np.argmax(policy.cpu().detach().data.numpy()).max()
        else:
            policy = policy.cpu().detach().numpy()
            action = np.random.choice(self.action_size, 1, p=policy[0]).max()
            # action = self.env.action_space[action[0]]
        return action

    def learn(self, experiences=None):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions, rewards, next_states, dones = self.memory.sample()

        policies, values = self.model(states)
        _, next_values = self.model(next_states)

        # rollout experiences and calculate expected value for each step
        target_vales = []
        R = next_values[-1]
        for idx in reversed(range(len(rewards))):
            R = rewards[idx] + self.GAMMA * R * (1.0 - dones[idx])
            # always add to the list at the first pos since we are going backwards
            target_vales.insert(0, R)

        log_probs = torch.log(policies)
        entropies = -(policies * log_probs).mean()

        advantages = torch.cat(target_vales).view(-1, 1) - values
        # TODO: actor loss detach advantage
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropies

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clean memeory buffer for the next trajectory
        self.memory.reset()


