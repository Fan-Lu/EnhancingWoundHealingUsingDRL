####################################################
# Description: Asynchronous Advantage Actor Critic
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-23
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from collections import deque
import sys

from utils.memories import ReplayBuffer
from utils.tools import SharedAdam

class A3C_Net(nn.Module):

    def __init__(self, state_size, action_size):
        super(A3C_Net, self).__init__()
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

        value = self.valueFunc(x)
        policy = F.softmax(self.policyFunc(x), dim=1)

        return value, policy


class Agent_A3C(object):

    def __init__(self, env, args):
        super(Agent_A3C, self).__init__()
        self.env = env
        self.args = args
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.LR = args.LR
        self.num_worker = args.num_worker

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

        # Global Model
        self.a3c_global = A3C_Net(self.state_size, self.action_size).to(self.device)
        # self.optimizer = optim.Adam(self.a3c_global.parameters(), lr=self.LR)
        self.optimizer = SharedAdam(self.a3c_global.parameters(), lr=self.LR)

    def step(self):
        workers = [Worker(self.a3c_global, self.optimizer, self.device, self.env, self.args)
                   for _ in range(self.num_worker)]

        # for gpu processing
        if torch.cuda.is_available() and self.args.gpu:
            mp.set_start_method('spawn')

        # Multi Processing
        process_buf = []
        for worker_idx in range(self.num_worker):
            worker = workers[worker_idx]
            process = mp.Process(target=worker.run, args=(worker_idx + 1,))
            process.start()
            process_buf.append(process)
        for p in process_buf:
            p.join()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.a3c_global.eval()
        with torch.no_grad():
            _, policy = self.a3c_global(state)
        self.a3c_global.train()

        return self.env.action_space[np.argmax(policy.cpu().data.numpy())]


class Worker(object):

    def __init__(self, a3c_global, optimizer, device, env, args):
        super(Worker).__init__()
        self.env = env
        self.args = args
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.batch_size = args.batch_size
        self.LR = args.LR
        self.GAMMA = args.GAMMA
        self.UPDATE_EVERY = args.UPDATE_EVERY
        self.actor_ratio = args.actor_ratio
        self.entropy_beta = args.entropy_beta

        self.device = device
        self.optimizer = optimizer
        self.a3c_global = a3c_global

        self.a3c_local = A3C_Net(self.state_size, self.action_size).to(self.device)
        # self.a3c_local.load_state_dict(self.a3c_global.state_dict())

        # Replay memory
        self.memory = ReplayBuffer(self.state_size, self.action_size, args.buffer_size, self.batch_size, self.device, args.seed)
        # Initialize time step (for updating every UPDATE_EVERY step)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.push(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn every UPDATE_EVERY steps
            if self.memory.__len__() > self.batch_size:
                # Off-policy training
                experiences = self.memory.sample()
                self.learn(experiences)

    def share_grad(self, local_model, target_model):
        '''
        Transfer the model's gradients to the target model
        '''
        for param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param._grad = param.grad

    def learn(self, experiences):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions, rewards, next_states, dones = experiences

        values, policies = self.a3c_local(states)
        next_values, _ = self.a3c_local(next_states)

        target_values = rewards + (self.GAMMA * next_values * (1 - dones))

        # Advantages
        advantages = target_values - values
        log_prob = torch.log(policies)
        loss_tmp = advantages * log_prob[range(actions.size(dim=1)), actions].view(-1, 1)

        # Calculate actor loss and critic losses
        act_loss = -loss_tmp.mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -(policies * log_prob).mean()
        loss = self.actor_ratio * act_loss + 0.5 * critic_loss + self.entropy_beta * entropy_loss

        # Optimize the global network
        self.optimizer.zero_grad()
        self.share_grad(self.a3c_local, self.a3c_global)
        loss.backward()
        self.optimizer.step()

        # load the network and reset memory
        self.a3c_local.load_state_dict(self.a3c_global.state_dict())

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.a3c_local.eval()
        with torch.no_grad():
            _, policy = self.a3c_local(state)
        self.a3c_local.train()

        policy = policy.cpu().detach().numpy()
        action = np.random.choice(self.action_size, 1, p=policy[0])
        return self.env.action_space[action[0]]

    def run(self, w_idx):
        writer = SummaryWriter()

        print('CheckOpt: {} ActionSize: {} UseGPU: {}'.format(self.args.check_opt, self.env.action_size, self.args.gpu))
        scores = []  # List containing scores from each episode
        scores_window = deque(maxlen=5)  # last 100 scores

        for i_episode in range(1, self.args.n_episodes + 1):
            state = self.env.reset()
            t = score = 0
            for t in range(self.args.max_t):
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                print('Worker: {}/{} \t Episode: {}/{} Step: {}/{} Action: {:.1f} NewTissue: {:.2f}'.format(
                    w_idx, self.args.num_worker, i_episode, self.args.n_episodes, t + 1, self.args.max_t, action, info[-1]), flush=True)
                sys.stdout.flush()
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)         # save most recent score
            scores.append(score)                # save most recent score
            # eps decayed to eps_end after around 1000 episodes
            writer.add_scalar('DaysTaken', self.env.t_span[t + 1] / self.args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
            print('Episode: {}/{}\tAverage Score: {:.2f}'.format(i_episode, self.args.n_episodes, np.mean(scores_window)), flush=True)
            sys.stdout.flush()
            if i_episode % 5 == 0:
                torch.save(self.a3c_global.state_dict(), './res/models_a3c/checkpoint_anum_{}_ep_{}_woker_{}.pth'.format(self.env.action_size, i_episode, w_idx))
        return scores