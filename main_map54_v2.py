####################################################
# Description: trainer of different algorithm
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-10-04
###################################################
import copy
import os
import random
import time
import sys

import numpy as np
import pandas as pd
from collections import deque

import matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from control import lqr

import torch
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter

from algs.ode54 import Transformer, Agent_Trans
from algs.a2c import Agent_A2C
from algs.dqn import Agent_DQN
from algs.ppo import Agent_PPO
from envs.env import WoundEnv, SimpleEnv, LinearEnv, dynamics, dynamics5, linear_dynamic
from cfgs.config import GetParameters

import torch.multiprocessing as mp

def linear_step(st_nl, st_tp1_nl, device, mapper, AA, BB):
    Q = np.eye(4)
    Q[3][3] = 0.0
    R = np.eye(5)
    st_nl_pos0 = torch.from_numpy(st_nl.reshape(5, args.n_cells)[:, 0]).float().to(device).view(1, -1)
    st_l, _, _ = mapper.model(st_nl_pos0)
    st_l = st_l.cpu().data.numpy().reshape(-1, 1)
    agent_state_next = st_tp1_nl.reshape(5, args.n_cells)[:, 0]

    try:
        K, S, E = lqr(AA, BB, Q, R)
        lact = -K @ st_l
        st_tp1_l = AA @ st_l + BB @ lact

        st_tp1_nl_ref = mapper.model.map425(torch.softmax(torch.from_numpy(st_tp1_l).view(1, -1), dim=1))
        st_tp1_nl_ref = st_tp1_nl_ref.cpu().data.numpy()

        reward_ref = (np.exp(-np.linalg.norm(st_tp1_nl_ref - agent_state_next)) - 1)
    except:
        reward_ref = -1.0

    return reward_ref


def map_trainer(alg_name, max_ep=500, args=None):
    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + alg_name
    runs_dir = args.model_dir + '../../runs_{}/{}'.format(alg_name, runs_dir)
    os.makedirs(runs_dir)

    writer = SummaryWriter(log_dir=runs_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    eps = args.eps_start
    # simulation of nonlinear sys
    nenv = WoundEnv(args)

    if not args.cont:
        agent = Agent_A2C(nenv, args)
    else:
        agent = Agent_PPO(nenv, args)
    mapper = Agent_Trans(nenv, args)

    Q = np.eye(4)
    Q[3][3] = 0.0
    R = np.eye(5)

    st_nl = nenv.reset()
    states_buf = copy.deepcopy(st_nl)
    for t in range(nenv.t_nums - 1):
        st_tp1_nl, reward_nl, done_nl, _ = nenv.step(0)
        st_nl = st_tp1_nl
        states_buf = np.vstack((states_buf, st_nl))

    args.ctr = False
    for i_episode in range(100):
        loss_mean, mse1_mean, mse2_mean, reward_mean = 0.0, 0.0, 0.0, 0.0
        d1 = time.time()
        for t in range(nenv.t_nums):
            st_nl = states_buf[t, :]
            loss_mean += mapper.step(st_nl, 0)
        if (i_episode+1) % 5 == 0:
            d2 = time.time()
            print('TrainEp_w/o_ctr: {} \t loss_total: {:.4f} Time: {:.2f} sec'.format(
                i_episode + 1, loss_mean / nenv.t_nums, d2 - d1), flush=True)
            sys.stdout.flush()

    args.ctr = True
    for i_episode in range(max_ep):
        st_nl = nenv.reset()
        loss_mean, mse1_mean, mse2_mean, reward_mean = 0.0, 0.0, 0.0, 0.0
        d1 = time.time()
        heal_day = None
        for t in range(nenv.t_nums - 1):
            st_nl_pos0 = torch.from_numpy(st_nl.reshape(5, args.n_cells)[:, 0]).float().to(device).view(1, -1)
            # selection action based using DRL
            agent_state = st_nl_pos0.cpu().data.numpy().squeeze()
            # if we do not need any control, set u equals 0
            action = agent.act(agent_state, eps) if args.ctr and heal_day is None else 0
            loss_mean += mapper.step(st_nl, action)

            # pass action into nonlinear system
            st_tp1_nl, reward_nl, done_nl, _ = nenv.step(action)
            # calculate the jajobian matrix

            '''
            For experiment without any input, ignore code from line 92 to line 110
            '''
            if args.ctr:
                jac = jacobian(mapper.model.map524, st_nl_pos0, create_graph=False)
                # m1 macrophage in the nonlinear dynamics
                m1 = st_nl_pos0.cpu().data.numpy()[:, 1][0]
                B = torch.from_numpy(np.array([[0,  0, 0, 0, 0],
                                               [0, -m1, 0, 0, 0],
                                               [0,  m1, 0, 0, 0],
                                               [0,  0, 0, 0, 0],
                                               [0,  0, 0, 0, 0]])).float().to(device)

                mapper.model.matmult(torch.zeros((1, 4)))
                AA = mapper.model.Amat_masked.cpu().data.numpy()
                BB = torch.matmul(jac[0, :, 0, :], B).cpu().data.numpy()

                st_l, _, _ = mapper.model(st_nl_pos0)
                st_l = st_l.cpu().data.numpy().reshape(-1, 1)
                agent_state_next = st_tp1_nl.reshape(5, args.n_cells)[:, 0]
                try:
                    K, S, E = lqr(AA, BB, Q, R)
                    lact = -K @ st_l
                    st_tp1_l = AA @ st_l + BB @ lact

                    st_tp1_nl_ref = mapper.model.map425(torch.softmax(torch.from_numpy(st_tp1_l).view(1, -1), dim=1))
                    st_tp1_nl_ref = st_tp1_nl_ref.cpu().data.numpy()

                    reward_ref = (np.exp(-np.linalg.norm(st_tp1_nl_ref - agent_state_next)) - 1)

                except:
                    reward_ref = -1.0

                reward = reward_ref + reward_nl

                if not done_nl:
                    agent.step(agent_state, action, reward, agent_state_next, done_nl)

                if reward_nl and heal_day is not None:
                    heal_day = nenv.t_days[t] / 3.0

                reward_mean += reward

            # state_next_non_linear, reward, done, info = env.step(u)
            st_nl = st_tp1_nl

        # loss = mapper.learn()
        # we only update RL agent when control is required
        if args.ctr:
            agent.learn()
        eps = max(args.eps_end, args.eps_decay * eps)  # decrease epsilon

        writer.add_scalar('Loss/train_mse', loss_mean / nenv.t_nums, i_episode)
        writer.add_scalar('DRL/Reward', reward_mean, i_episode)
        writer.add_scalar('DRL/Eps', eps, i_episode)
        writer.add_scalar('Loss/Lamda', mapper.lam, i_episode)

        writer.add_scalar('Loss/K_h', mapper.model.Amat_masked.cpu().data.numpy()[1, 0], i_episode)
        writer.add_scalar('Loss/K_i', mapper.model.Amat_masked.cpu().data.numpy()[2, 1], i_episode)
        writer.add_scalar('Loss/K_p', mapper.model.Amat_masked.cpu().data.numpy()[3, 2], i_episode)

        if (i_episode+1) % 5 == 0:
            heal_day = test(colab_dir, device, (mapper.model, agent), writer, i_episode, args)
            d2 = time.time()
            print('TrainEpCtr: {} \t RewardMean: {:.4f} loss_total: {:.4f} HealDay: {:.2f} Time: {:.2f} sec'.format(
                i_episode + 1, reward_mean, loss_mean / nenv.t_nums, heal_day, d2 - d1), flush=True)
            sys.stdout.flush()

            torch.save(mapper.model.state_dict(),
                       args.model_dir + 'mapper_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))
            if args.ctr:
                torch.save(agent.model.state_dict(),
                           args.model_dir + 'agent_ctr_{}_ep_{}.pth'.format(args.ctr, i_episode))
            # record the testing healday
            writer.add_scalar('DRL/HealDay', heal_day, i_episode)


def test(colab_dir, device, models, writer, i_episode, args):

    nenv = WoundEnv(args)

    state_non_linear = nenv.reset()
    state_linear_chainrule_buf, state_linear_linearapx_buf, state_non_linear_buf, state_non_linear_est_buf = [], [], [], []
    cstate_linear_buf = []
    action_buf = []

    model, agent = models

    heal_day = None

    # print(model.Amat_masked.cpu().data.numpy())
    for t in range(nenv.t_nums - 1):

        new_tissue = state_non_linear.reshape(5, args.n_cells)[4, 0]

        if new_tissue > 0.95 and heal_day is None:
            heal_day = nenv.t_span[t] / 3.0

        state_non_linear_buf.append(state_non_linear.reshape(5, args.n_cells))
        state_non_linear_5 = state_non_linear.reshape(5, args.n_cells)[:, 0]
        state_non_linear_tensor = torch.from_numpy(state_non_linear_5).float().to(device).view(1, -1)
        # state_non_linear_tensor[:, 3] /= 3.0

        # selection action based using DRL
        u = agent.act(state_non_linear_tensor.cpu().data.numpy().squeeze()) if args.ctr and heal_day is None else 0
        state_next_non_linear, reward_2_abandon, done_2_abandon, info = nenv.step(u)

        state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
        state_non_linear_est_buf.append(state_non_linear_aprx.cpu().data.numpy().reshape(5, -1))

        jac = jacobian(model.map524, state_non_linear_tensor, create_graph=True)

        n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
        arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
        dxdt = dynamics5(state_non_linear, None, u, arrgs)
        dxdt_5 = dxdt.reshape(5, args.n_cells)[:, 0]
        Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt_5).float().to(device).view(-1, 1)).view(1, -1)

        # m1 macrophage
        v = state_non_linear_tensor.cpu().data.numpy()[:, 1][0]
        B = torch.from_numpy(np.array([[0, 0, 0, 0, 0],
                                       [0, -v, 0, 0, 0],
                                       [0, v, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]])).float().to(device)
        u_tensor = torch.from_numpy(np.array([u, u, u, u, u])).float().to(device).view(-1, 1)
        Jn_bu = torch.matmul(jac[0, :, 0, :], torch.matmul(B, u_tensor)).view(1, -1)

        react = nenv.theta_space[u]
        action_buf.append(react)

        cstate_linear_buf.append(state_linear.cpu().data.numpy())
        state_linear_chainrule_buf.append(Jn_dxdt.cpu().data.numpy())
        state_linear_linearapx = state_linear_AN + Jn_bu
        state_linear_linearapx_buf.append(state_linear_linearapx.cpu().data.numpy())

        state_non_linear = state_next_non_linear

    cstate_linear_buf = np.array(cstate_linear_buf).squeeze()
    state_linear_chainrule_buf = np.array(state_linear_chainrule_buf).squeeze()
    state_linear_linearapx_buf = np.array(state_linear_linearapx_buf).squeeze()
    state_non_linear_buf = np.array(state_non_linear_buf).squeeze()
    state_non_linear_est_buf = np.array(state_non_linear_est_buf).squeeze()
    action_buf = np.array(action_buf).squeeze()
    trange = (nenv.t_span / 3.0)[:-1]

    im_scal = 0.8
    leg_pos = (1, 0.5)
    fig = plt.figure(figsize=(8, 12), num=4)
    plt.tight_layout()

    ax = fig.add_subplot(411)
    ax.plot(trange, action_buf, label='Actualtion')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(412)
    ax.plot(trange, cstate_linear_buf[:, 0], color='r', linestyle='--', label=r'$H-trans$')
    ax.plot(trange, cstate_linear_buf[:, 1], color='g', linestyle='--', label=r'$I-trans$')
    ax.plot(trange, cstate_linear_buf[:, 2], color='b', linestyle='--', label=r'$P-trans$')
    ax.plot(trange, cstate_linear_buf[:, 3], color='y', linestyle='--', label=r'$M-trans$')
    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(413)
    ax.plot(trange, state_linear_chainrule_buf[:, 0], color='r', linestyle='--', label=r'$\dot{H}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 1], color='g', linestyle='--', label=r'$\dot{I}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 2], color='b', linestyle='--', label=r'$\dot{P}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 3], color='y', linestyle='--', label=r'$\dot{M}-chain$')

    ax.plot(trange, state_linear_linearapx_buf[:, 0], color='r', linestyle='-', label=r'$\dot{H}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 1], color='g', linestyle='-', label=r'$\dot{I}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 2], color='b', linestyle='-', label=r'$\dot{P}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 3], color='y', linestyle='-', label=r'$\dot{M}-laprx$')

    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(414)
    ax.plot(trange, state_non_linear_buf[:, 0, 0], color='r', linestyle='--', label='Debris-ref')
    ax.plot(trange, state_non_linear_buf[:, 1, 0], color='g', linestyle='--', label='M1-ref')
    ax.plot(trange, state_non_linear_buf[:, 2, 0], color='b', linestyle='--', label='M2-ref')
    ax.plot(trange, state_non_linear_buf[:, 3, 0] / args.nscale, color='y', linestyle='--', label='Temp-ref')
    ax.plot(trange, state_non_linear_buf[:, 4, 0], color='c', linestyle='--', label='New-ref')

    ax.plot(trange, state_non_linear_est_buf[:, 0], color='r', linestyle='-', label='Debris-est')
    ax.plot(trange, state_non_linear_est_buf[:, 1], color='g', linestyle='-', label='M1-est')
    ax.plot(trange, state_non_linear_est_buf[:, 2], color='b', linestyle='-', label='M2-est')
    ax.plot(trange, state_non_linear_est_buf[:, 3] / args.nscale, color='y', linestyle='-', label='Temp-est')
    ax.plot(trange, state_non_linear_est_buf[:, 4], color='c', linestyle='-', label='New-est')

    ax.set_xlabel('time, day')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    # plt.show()
    writer.add_figure('Wound/Test_ctr_{}'.format(args.ctr), fig, i_episode)
    plt.close()

    return heal_day


def map_trainer_noctr(alg_name, max_ep=500, args=None):
    # directories checking and creation

    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + alg_name + '_sd_{}'.format(args.seed)
    runs_dir = args.model_dir + '../../runs_{}/{}'.format(alg_name, runs_dir)
    os.makedirs(runs_dir)

    # tensorboard writer to display all the data
    writer = SummaryWriter(log_dir=runs_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    eps = args.eps_start
    nenv = WoundEnv(args)

    st_nl = nenv.reset()
    states_buf = copy.deepcopy(st_nl)
    for t in range(nenv.t_nums - 1):
        st_tp1_nl, reward_nl, done_nl, _ = nenv.step(0)
        st_nl = st_tp1_nl
        states_buf = np.vstack((states_buf, st_nl))

    mapper = Agent_Trans(nenv, args)

    for i_episode in range(max_ep):
        d1 = time.time()
        loss_mean, mse1_mean, mse2_mean, reward_mean = 0.0, 0.0, 0.0, 0.0
        for t in range(nenv.t_nums):
            st_nl = states_buf[t, :]
            loss_mean += mapper.step(st_nl, 0)

        eps = max(args.eps_end, args.eps_decay * eps)  # decrease epsilon

        writer.add_scalar('Loss/train_mse', loss_mean / nenv.t_nums, i_episode)
        writer.add_scalar('Loss/Lamda', mapper.lam, i_episode)

        writer.add_scalar('Loss/K_h', mapper.model.Amat_masked.cpu().data.numpy()[1, 0], i_episode)
        writer.add_scalar('Loss/K_i', mapper.model.Amat_masked.cpu().data.numpy()[2, 1], i_episode)
        writer.add_scalar('Loss/K_p', mapper.model.Amat_masked.cpu().data.numpy()[3, 2], i_episode)

        if (i_episode+1) % 5 == 0:
            heal_day = test_noctr(device, (mapper.model, states_buf), writer, i_episode, args)
            d2 = time.time()
            print('Seed: {} TrainEp: {} \t RewardMean: {:.4f} loss_total: {:.4f} HealDay: {:.2f} Time: {:.2f} sec'.format(
                args.seed, i_episode + 1, reward_mean / nenv.t_nums, loss_mean / nenv.t_nums, heal_day, d2 - d1))
            torch.save(mapper.model.state_dict(),
                       args.model_dir + 'checkpoint_ctr_{}_ep_{}_seed{}.pth'.format(args.ctr, i_episode, args.seed))


def test_noctr(device, models, writer, i_episode, args):

    nenv = WoundEnv(args)

    state_linear_chainrule_buf, state_linear_linearapx_buf, state_non_linear_buf, state_non_linear_est_buf = [], [], [], []
    cstate_linear_buf = []
    action_buf = []

    model, state_buf = models

    heal_day = None

    # print(model.Amat_masked.cpu().data.numpy())
    for t in range(nenv.t_nums - 1):
        state_non_linear = state_buf[t, :]
        new_tissue = state_non_linear.reshape(5, args.n_cells)[4, 0]
        if new_tissue > 0.95 and heal_day is None:
            heal_day = nenv.t_span[t] / 3.0

        state_non_linear_buf.append(state_non_linear.reshape(5, args.n_cells))
        state_non_linear_5 = state_non_linear.reshape(5, args.n_cells)[:, 0]
        state_non_linear_tensor = torch.from_numpy(state_non_linear_5).float().to(device).view(1, -1)

        state_linear, state_linear_AN, state_non_linear_aprx = model(state_non_linear_tensor)
        state_non_linear_est_buf.append(state_non_linear_aprx.cpu().data.numpy().reshape(5, -1))

        jac = jacobian(model.map524, state_non_linear_tensor, create_graph=False)

        n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n = args.n_cells, args.spt, args.X_pump, args.beta, args.gamma1, args.gamma2, args.rho, args.mu, args.alphaTilt, args.power, args.kapa, args.Lam, args.DTilt, args.DTilt_n
        arrgs = (n_cells, spt, X_pump, beta, gamma1, gamma2, rho, mu, alphaTilt, power, kapa, Lam, DTilt, DTilt_n)
        dxdt = dynamics5(state_non_linear, None, 0, arrgs)
        dxdt_5 = dxdt.reshape(5, args.n_cells)[:, 0]
        Jn_dxdt = torch.matmul(jac[0, :, 0, :], torch.from_numpy(dxdt_5).float().to(device).view(-1, 1)).view(1, -1)

        action_buf.append(nenv.theta_space[0])

        cstate_linear_buf.append(state_linear.cpu().data.numpy())
        state_linear_chainrule_buf.append(Jn_dxdt.cpu().data.numpy())
        state_linear_linearapx = state_linear_AN
        state_linear_linearapx_buf.append(state_linear_linearapx.cpu().data.numpy())


    cstate_linear_buf = np.array(cstate_linear_buf).squeeze()
    state_linear_chainrule_buf = np.array(state_linear_chainrule_buf).squeeze()
    state_linear_linearapx_buf = np.array(state_linear_linearapx_buf).squeeze()
    state_non_linear_buf = np.array(state_non_linear_buf).squeeze()
    state_non_linear_est_buf = np.array(state_non_linear_est_buf).squeeze()
    action_buf = np.array(action_buf).squeeze()
    trange = (nenv.t_span / 3.0)[:-1]

    im_scal = 0.8
    leg_pos = (1, 0.5)
    fig = plt.figure(figsize=(8, 12), num=4)
    plt.tight_layout()

    ax = fig.add_subplot(411)
    ax.plot(trange, action_buf, label='Actualtion')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(412)
    ax.plot(trange, cstate_linear_buf[:, 0], color='r', linestyle='--', label=r'$H-trans$')
    ax.plot(trange, cstate_linear_buf[:, 1], color='g', linestyle='--', label=r'$I-trans$')
    ax.plot(trange, cstate_linear_buf[:, 2], color='b', linestyle='--', label=r'$P-trans$')
    ax.plot(trange, cstate_linear_buf[:, 3], color='y', linestyle='--', label=r'$M-trans$')
    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(413)
    ax.plot(trange, state_linear_chainrule_buf[:, 0], color='r', linestyle='--', label=r'$\dot{H}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 1], color='g', linestyle='--', label=r'$\dot{I}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 2], color='b', linestyle='--', label=r'$\dot{P}-chain$')
    ax.plot(trange, state_linear_chainrule_buf[:, 3], color='y', linestyle='--', label=r'$\dot{M}-chain$')

    ax.plot(trange, state_linear_linearapx_buf[:, 0], color='r', linestyle='-', label=r'$\dot{H}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 1], color='g', linestyle='-', label=r'$\dot{I}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 2], color='b', linestyle='-', label=r'$\dot{P}-laprx$')
    ax.plot(trange, state_linear_linearapx_buf[:, 3], color='y', linestyle='-', label=r'$\dot{M}-laprx$')

    # ax.set_xlabel('time, sec')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    ax = fig.add_subplot(414)
    ax.plot(trange, state_non_linear_buf[:, 0, 0], color='r', linestyle='--', label='Debris-ref')
    ax.plot(trange, state_non_linear_buf[:, 1, 0], color='g', linestyle='--', label='M1-ref')
    ax.plot(trange, state_non_linear_buf[:, 2, 0], color='b', linestyle='--', label='M2-ref')
    ax.plot(trange, state_non_linear_buf[:, 3, 0] / 3.0, color='y', linestyle='--', label='Temp-ref')
    ax.plot(trange, state_non_linear_buf[:, 4, 0], color='c', linestyle='--', label='New-ref')

    ax.plot(trange, state_non_linear_est_buf[:, 0], color='r', linestyle='-', label='Debris-est')
    ax.plot(trange, state_non_linear_est_buf[:, 1], color='g', linestyle='-', label='M1-est')
    ax.plot(trange, state_non_linear_est_buf[:, 2], color='b', linestyle='-', label='M2-est')
    ax.plot(trange, state_non_linear_est_buf[:, 3] / 3.0, color='y', linestyle='-', label='Temp-est')
    ax.plot(trange, state_non_linear_est_buf[:, 4], color='c', linestyle='-', label='New-est')

    ax.set_xlabel('time, day')
    ax.legend(loc='center left', bbox_to_anchor=leg_pos)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * im_scal, box.height])

    # plt.show()
    writer.add_figure('Wound/Test_ctr_{}'.format(args.ctr), fig, i_episode)

    # plt.savefig(args.figs_dir + '')
    plt.close()

    return heal_day


if __name__ == "__main__":
    args = GetParameters()
    colab_dir = "../../../ExpDataDARPA/"

    alg_name = 'map524'
    args.model_dir = colab_dir + '/res_map524/models_{}/'.format(alg_name)
    args.data_dir = colab_dir + '/res_map524/data_{}/'.format(alg_name)
    args.figs_dir = colab_dir + '/res_map524/figs_{}/'.format(alg_name)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    if args.ctr:
        map_trainer(alg_name, max_ep=10000, args=args)
    else:

        # test 100 independent runs
        for sd in range(5, 20):
            jobs = []
            for pd in range(5):
                args.seed = sd * 5 + pd
                p = mp.Process(target=map_trainer_noctr, args=(alg_name, 10000, args, ))
                jobs.append(p)
                p.start()
                time.sleep(0.1)

            for pp in jobs:
                time.sleep(0.1)
                pp.join()

        # map_trainer_noctr(colab_dir, max_ep=10000, args=args)






