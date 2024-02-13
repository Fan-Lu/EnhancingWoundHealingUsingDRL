####################################################
# Description: trainer of different algorithm
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-09-01
###################################################

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

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from algs.dqn import Agent_DQN
from algs.a3c import Agent_A3C
from algs.a2c import Agent_A2C
from algs.ppo import Agent_PPO
from algs.td3 import Agent_TD3
from envs.env import WoundEnv, SimpleEnv

def mesh_2d_mat(t, r, tdays, radius, vector):
    @np.vectorize
    def value(tval, rval):
        tidx = np.where(t == tval)[0][0]
        ridx = np.where(r == rval)[0][0]
        val = vector[tidx][ridx]
        return val

    valuefun = value(tdays, radius)
    return valuefun


def healing_time(new_tissues):
    wound_radius_list = []
    r = np.linspace(0, 3, 100)
    for t_idx in range(601):
        new_tissue_t_day = new_tissues[t_idx, :]
        rmin = 99
        for ridx in range(100):
            if new_tissue_t_day[ridx] >= 0.95:
                rmin = ridx
                break
        wound_radius = r[rmin]
        wound_radius_list.append(wound_radius)
    return wound_radius_list


def plotsRes(args, model_idx):
    t = np.linspace(0, 60, 601)
    r = np.linspace(0, 3, 100)
    t_ode = t / 6.0 + 1.2
    tdays, radius = np.meshgrid(t, r)

    r = np.linspace(0, 3, 100)
    # states_act0_noRL, states_act1_noRL, states_RL = vectors

    states_act0_noRL_DF = pd.read_csv(args.data_dir + 'state_a0_info.csv')
    # states_act1_noRL_DF = pd.read_csv(args.data_dir + 'state_a1_info.csv')
    states_RL_DF = pd.read_csv(args.data_dir + 'state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))
    actions_RL_DF = pd.read_csv(args.data_dir + 'action_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    states_act0_noRL = states_act0_noRL_DF.values[:, 100 * 4 + 1:]
    # states_act1_noRL = states_act1_noRL_DF.values[:, 100 * 4 + 1:]
    states_RL = states_RL_DF.values[:, 100 * 4 + 1:]
    actions_RL = actions_RL_DF.values[:, 1:]

    new_at_all_r = states_RL_DF.values[:, 100 * 4 + 1:]

    wound_radius_list_a0 = healing_time(states_act0_noRL)
    # wound_radius_list_a1 = healing_time(states_act1_noRL)
    wound_radius_list_rl = healing_time(states_RL)

    fig = plt.figure(figsize=(12, 4), num=3)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(t[1:] / 3.0, actions_RL)
    ax.set_title('RL Policies')

    valuefun = mesh_2d_mat(t, r, tdays, radius, new_at_all_r)
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_surface(radius, tdays, valuefun, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_title(r'New Tissue with RL Treatment')
    ax.set_xlabel(r'radius $r$, mm')
    ax.set_ylabel(r't, days')
    # ax.set_zlabel(r'New Tissue', loc='left')

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t / 3.0, wound_radius_list_a0, color='g', linestyle='--', label='No Treatment')
    # ax.plot(t / 3.0, wound_radius_list_a1, color='r', linestyle='-', label='ODE Treatment')
    ax.plot(t / 3.0, wound_radius_list_rl, color='b', linestyle='-', label='RL Treatment')
    ax.plot(t_ode, wound_radius_list_a0, color='m', linestyle='--', label='Twice of No Treatment')
    ax.set_xlabel(r't, days')
    ax.set_ylabel(r'wound radius, mm')
    ax.set_title('Wound Radius')
    ax.legend()

    fig.tight_layout()
    plt.savefig(args.figs_dir + 'test_all_rl_{}_anum_{}_ep{}.pdf'.format(args.alg_rl, args.action_size, model_idx))
    plt.close()


def apply_noRL_treatment(args, action):
    args.check_opt = True
    env = WoundEnv(args)
    state = env.reset()
    state_buf, action_buf = [], []
    k = 0
    state_buf.append(state)
    for k in range(600):
        next_state, reward, done, info = env.step(action)
        print('Time: {} Action: {} New Tissue: {:.2f}'.format(k + 1, action, info[-1][-1]))
        state = next_state
        state_buf.append(state)
    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    if len(action) < 2:
        state_buf_df.to_csv(args.data_dir + 'state_a{}_info.csv'.format(0))
    else:
        state_buf_df.to_csv(args.data_dir + 'state_a_{}_{}_info.csv'.format(0, args.X_pump))

    return env.t_span[k + 1] / 3.0


def apply_RL_treatment(args, model_idx, agent, env):
    args.check_opt = False

    agent.model.load_state_dict(
        torch.load(args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(env.action_space.shape[0], model_idx)))
    state = env.reset()
    state_buf, action_buf = [], []
    state_buf.append(state)
    k = 0
    for k in range(600):
        if state[4 * env.n_cells] < 0.95 and k >= 3:
            action = agent.act(state, 0)
        else:
            if args.spt:
                action = np.array([0.0, 80])
            else:
                action = 0.0
        next_state, reward, done, info = env.step(action)

        state = next_state
        state_buf.append(state)
        action_buf.append(action)
        if not args.spt:
            print('Alg: {} Ep: {} T: {} Act: {:.2f} NT: {:.2f}'.format(
                args.alg_rl, model_idx, k + 1, action, info[4][0]))
        else:
            print('Alg: {} Ep: {} T: {} Act: [{:.2f} {}] NT: {:.2f}'.format(
                args.alg_rl, model_idx, k + 1, action[0], int(action[1]), info[4][0]))

    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    state_buf_df.to_csv(args.data_dir + 'state_rl_anum_{}_ep_{}.csv'.format(env.action_space.shape[0], model_idx))

    action_buf = np.array(action_buf)
    action_buf_df = pd.DataFrame(action_buf)
    action_buf_df.to_csv(args.data_dir + 'action_rl_anum_{}_ep_{}.csv'.format(env.action_space.shape[0], model_idx))

    return env.t_span[k + 1] / 3.0


def a3c_trainer(env, args, writer=None):
    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter()
    model_dir = './res/models_a3c/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    agent = Agent_A3C(env, args)
    agent.step()
    return agent


def dqn_trainer(env, args, writer=None):
    agent = Agent_DQN(env, args)

    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    eps = args.eps_start
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        t = score = 0
        for t in range(args.t_nums - 1):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            print('Episode: {}/{} Step: {}/{}\tAction: {:.1f} NewTissue: {:.2f}'.format(
                i_episode, args.n_episodes, t + 1, args.t_nums, action, info[-1]), flush=True)
            sys.stdout.flush()
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        # eps decayed to eps_end after around 1000 episodes
        eps = max(args.eps_end, args.eps_decay * eps) # decrease epsilon
        if writer is not None:
            writer.add_scalar('DaysTaken', env.t_span[t + 1] / args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
            writer.add_scalar('Epsilon', eps, i_episode)
        print('Algorithm: {} Episode: {}/{}\tAverage Score: {:.2f}'.format(args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window)), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.model.state_dict(), args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
    return scores


def a2c_trainer(env, args, writer=None):
    agent = Agent_A2C(env, args)
    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        score = 0
        t1 = time.time()
        while env.cnter <= args.t_nums:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # print('Episode: {}/{} Step: {}/{}\tTheta: {:.4f} NewTissue: {:.2f}'.format(
            #     i_episode, args.n_episodes, env.cnter, args.t_nums, env.theta_space[action], info[-1][0]), flush=True)
            sys.stdout.flush()
            state = next_state
            score += reward
            if done:
                break
        # Episode Update
        agent.learn()
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        if writer is not None:
            writer.add_scalar('DaysTaken', env.t_span[env.cnter] / args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
        t2 = time.time()
        print('Algorithm: {} Episode: {}/{}\t HealDay: {:.2f} Time: {:.2f}'.format(
            args.alg_rl, i_episode, args.n_episodes, env.t_span[env.cnter] / args.Tc, t2 - t1), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, i_episode))
    return scores


def ppo_trainer(env, args, writer=None):
    agent = Agent_PPO(env, args)

    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        t = score = 0
        for t in range(env.t_nums - 1):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # if not args.spt:
            #     print('Episode: {}/{} Step: {}/{}\tTheta: {:.3f} MaxInflam NewTissue: {:.2f}'.format(
            #         i_episode, args.n_episodes, env.cnter, args.t_nums, action[0], info[-1][0]), flush=True)
            # else:
            #     print('Episode: {}/{} Step: {}/{}\tTheta: {:.3f} Pos: {} NewTissue: {:.2f}'.format(
            #         i_episode, args.n_episodes, env.cnter, args.t_nums, action[0], action[1], info[-1][0]), flush=True)
            sys.stdout.flush()

            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        if writer is not None:
            writer.add_scalar('DaysTaken', env.t_span[t + 1] / args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
            writer.add_scalar('ActionSTD', agent.action_std, i_episode)
        # print('Algorithm: {} Episode: {}/{}\tAverage Score: {:.2f}'.format(
        #     args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window)), flush=True)
        # print('Algm: {} Ep: {}/{}\tAvgS: {:.2f}, info: [M: {:.2f} Stage: {}] Action: [{:.2f}, {:.2f}, {:.2f}]'.format(
        #     args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window),
        #     info[0][3], info[1], action[0], action[1], action[2]), flush=True)
        print('Episode: {}/{} Step: {}/{}\tTheta: {:.3f} Pos: {} NewTissue: {:.2f}'.format(
            i_episode, args.n_episodes, env.cnter, args.t_nums, action[0], action[1], info[-1][0]), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(env.action_space.shape[0], i_episode))
    return scores


def td3_trainer(env, args, writer=None):
    agent = Agent_TD3(env, args)

    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    eps = args.eps_start
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        t = score = 0
        for t in range(args.t_nums - 1):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # print('Episode: {}/{} Step: {}/{}\tAction: {} NewTissue: {:.2f}'.format(
            #     i_episode, args.n_episodes, t + 1, args.max_t, action, info[-1][0]), flush=True)
            sys.stdout.flush()
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)         # save most recent score
        scores.append(score)                # save most recent score
        # eps decayed to eps_end after around 1000 episodes
        # eps = max(args.eps_end, args.eps_decay * eps) # decrease epsilon
        if writer is not None:
            writer.add_scalar('DaysTaken', env.t_span[t + 1] / args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
            writer.add_scalar('Epsilon', eps, i_episode)
            # writer.add_scalar('kh', action[0], i_episode)
            # writer.add_scalar('ki', action[1], i_episode)
            # writer.add_scalar('kp', action[2], i_episode)
        print('Algm: {} Ep: {}/{}\tAvgS: {:.2f}, info: [M: {:.2f} Stage: {}] Action: [{:.2f}, {:.2f}, {:.2f}]'.format(
            args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window), info[0][3], info[1], action[0], action[1], action[2]), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.actor.state_dict(),
                       args.model_dir + 'checkpoint_actor_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
            torch.save(agent.critic.state_dict(),
                       args.model_dir + 'checkpoint_critic_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
    return scores
