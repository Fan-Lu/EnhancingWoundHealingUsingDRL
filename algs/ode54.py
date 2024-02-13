####################################################
# Description: Non-linear to linear dynamics
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-03
####################################################

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import jacobian


from utils.memories import TransCache
from envs.env import dynamics, dynamics5


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, nscale):
        '''
        Find mapping between non-linear and linear models
        '''
        super(Transformer, self).__init__()

        # torch.manual_seed(args.seed)
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc11 = nn.Linear(self.in_dim, 64)
        self.fc12 = nn.Linear(64, 128)
        self.fc13 = nn.Linear(128, 64)
        self.fc14 = nn.Linear(64, self.out_dim)

        self.fc21 = nn.Linear(self.out_dim, 64)
        self.fc22 = nn.Linear(64, 128)
        self.fc23 = nn.Linear(128, 64)
        self.fc24 = nn.Linear(64, self.in_dim)

        # self.Amat = nn.Linear(out_dim, out_dim, bias=False)
        kh, ki, kp = 0.1, 0.1, 0.1
        AInit = np.array([[kh, 0, 0, 0],
                          [kh, ki, 0, 0],
                          [0, ki, kp, 0],
                          [0, 0, kp, 0]])
        AMask = np.array([[-1, 0, 0, 0],
                          [1, -1, 0, 0],
                          [0, 1, -1, 0],
                          [0, 0, 1, 0]])

        self.outMap = torch.from_numpy(np.array([1, 1, 1, nscale, 1])).float().view(1, -1)
        self.AMask = torch.from_numpy(AMask).float()
        self.Amat = nn.Parameter(torch.from_numpy(AInit).float(), requires_grad=True)
        self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)

    def map524(self, x):
        z = F.relu(self.fc11(x))
        z = F.relu(self.fc12(z))
        z = F.relu(self.fc13(z))
        z = F.softmax(self.fc14(z), dim=1)
        return z

    def map425(self, z):
        x_hat = F.relu(self.fc21(z))
        x_hat = F.relu(self.fc22(x_hat))
        x_hat = F.relu(self.fc23(x_hat))
        x_hat = F.sigmoid(self.fc24(x_hat))

        x_hat = torch.multiply(x_hat, self.outMap)

        return x_hat

    def matmult(self, z):
        # self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        # Az = torch.matmul(self.Amat_masked, x2z.T)
        # self.Amat_masked = torch.clamp(torch.multiply(F.relu(self.Amat), self.AMask), min=0.01, max=1.0)
        self.Amat_masked = torch.multiply(F.relu(self.Amat), self.AMask)
        Az = torch.matmul(self.Amat_masked, z.T).T

        return Az

    def forward(self, x):
        z = self.map524(x)
        x_hat = self.map425(z)
        AN = self.matmult(z)

        return z, AN, x_hat


class Agent_Trans(object):

    def __init__(self, env, args):
        '''
        Advantage Actor Critic
        '''
        super(Agent_Trans, self).__init__()

        self.env = env
        self.args = args

        torch.manual_seed(self.args.seed)

        if not self.args.cloose_loop:
            self.state_size = env.state_size
        else:
            # TODO: To determined by the transformer
            self.state_size = args.decoder_size
        if not self.args.spt:
            if self.args.cont:
                self.action_size = env.action_space.shape[0]
            else:
                self.action_size = env.action_space.n
        else:
            self.action_size = env.action_space.shape[0]
        self.action_dim = 1
        # self.batch_size = self.args.batch_size
        self.LR = self.args.LR
        self.TAU = self.args.TAU
        self.GAMMA = self.args.GAMMA

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

        self.model = Transformer(in_dim=5, out_dim=4, nscale=args.nscale).to(self.device)
        # TODO: Integrated into transformer
        self.model.AMask = self.model.AMask.to(self.device)
        self.model.outMap = self.model.outMap.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        # self.optimizer = optim.Adadelta(self.model.parameters())
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ent_loss = nn.CrossEntropyLoss()
        self.log_loss = LogCoshLoss()
        self.hub_loss = nn.HuberLoss(delta=1e-1, reduction='none')
        self.mae_loss = nn.L1Loss(reduction="none")
        weights = torch.tensor([0.2, 0.2, 0.2, 0.05, 0.35])
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weights)

        self.gamma = 0.9995
        self.lam = 0.99
        self.min_lam = 0.2

        self.batch_size = 4

        # Replay memory
        # self.memory = TransCache(batch_size=self.env.t_nums - 1, device=self.device)
        self.memory = TransCache(batch_size=self.batch_size, device=self.device)

        self.rt_loss = 0.0

    def step(self, state, action):
        '''

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        '''

        # save experience in replay memory
        # self.memory.push(state, action, reward, next_state, done)
        self.memory.push(state, action)
        if self.memory.__len__() >= self.batch_size:
            self.rt_loss = self.learn()
        return self.rt_loss

    def learn(self):
        '''
        Update value parameters using given batch of experiences tuples

        :param experiences:
        :return:
        '''
        states, actions = self.memory.sample()

        batch_loss = []
        for t in range(self.memory.__len__()):
            # current state obtained from non-linear dynamics
            st_nl = states[t, :].view(1, -1)
            action = actions[t, :].cpu().data.numpy().max()
            # only key the center position
            st_nl_p0 = st_nl.view(5, self.args.n_cells)[:, 0].view(1, -1)

            st_l, st_l_AN, st_nl_apprx = self.model(st_nl_p0)
            jac = jacobian(self.model.map524, st_nl_p0, create_graph=True)

            # TODO: Fix config bug in Numba
            n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = self.args.n_cells, self.args.spt, self.args.X_pump, self.args.beta, self.args.gamma1, self.args.gamma2, self.args.rho, self.args.mu, self.args.alphaTilt, self.args.power, self.args.kapa, self.args.Lam, self.args.DTilt, self.args.DTilt_n
            arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
            dxdt = dynamics5(st_nl.cpu().data.numpy().reshape(-1), None, action, arrgs)

            # Get ODE at wound center
            dxdt_5 = torch.from_numpy(dxdt.reshape(5, self.args.n_cells)[:, 0]).float().to(self.device).view(-1, 1)
            Jn_dxdt_5 = torch.matmul(jac[0, :, 0, :], dxdt_5).view(1, -1)

            # m1 macrophage
            m1 = st_nl_p0.cpu().data.numpy()[:, 1][0]
            B = torch.from_numpy(np.array([[0,   0, 0, 0, 0],
                                           [0, -m1, 0, 0, 0],
                                           [0,  m1, 0, 0, 0],
                                           [0,   0, 0, 0, 0],
                                           [0,   0, 0, 0, 0]])
                                 ).float().to(self.device)
            u_tensor = torch.from_numpy(np.array([action, action, action, action, action])).float().to(self.device).view(-1, 1)
            Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor)).view(1, -1)

            # loss for ode approximation
            mse1 = self.mse_loss(Jn_dxdt_5, st_l_AN + Jn_bu)
            # regularization loss for decoder
            mse2 = self.mse_loss(st_nl_apprx, st_nl_p0)
            # mse2 = (mse2 * torch.tensor([1.0, 1.0, 1.0, 1.0/3.0, 1.0]))
            # bio interpretation loss
            mse3 = self.mse_loss(st_l[:, :3], st_nl_p0[:, :3])

            loss = mse1.mean() + self.lam * (mse2.mean() + mse3.mean())

            batch_loss.append(loss)

        self.lam = max(self.min_lam, self.lam * self.gamma)
        batch_loss = torch.vstack(batch_loss).view(self.memory.__len__(), -1).mean()
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        # Clean memeory buffer for the next trajectory
        self.memory.reset()

        return batch_loss.cpu().data.numpy().max()

