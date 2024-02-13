####################################################
# Description: Main
# Version: V0.0.0
# Author: Fan Lu @ UCSC
# Data: 2023-07-12
# Architecture of Directories
'''
-- codePy
    -- algs                 # Algorithms
    -- cfgs                 # configuration files
    -- envs                 # environments
    -- res                  # experimental results
        -- data             # data saved
        -- figs             # figures
        -- models           # trained models
'''
###################################################

import os
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
from envs.env import SimpleEnv
from cfgs.config_simple import GetParameters


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
    t_ode = t / 6.0 + 1.2

    r = np.linspace(0, 3, 100)
    # states_act0_noRL, states_act1_noRL, states_RL = vectors

    states_act0_noRL_DF = pd.read_csv(args.data_dir + 'state_a0_info.csv')
    states_act1_noRL_DF = pd.read_csv(args.data_dir + 'state_a1_info.csv')
    states_RL_DF = pd.read_csv(args.data_dir + 'state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))
    actions_RL_DF = pd.read_csv(args.data_dir + 'action_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    states_act0_noRL_YY1 = states_act0_noRL_DF.values[:, 100 * 1 + 1]
    states_act1_noRL_YY1 = states_act1_noRL_DF.values[:, 100 * 1 + 1]
    states_RL_YY1 = states_RL_DF.values[:, 100 * 1 + 1]

    states_act0_noRL_YY2 = states_act0_noRL_DF.values[:, 100 * 2 + 1]
    states_act1_noRL_YY2 = states_act1_noRL_DF.values[:, 100 * 2 + 1]
    states_RL_YY2 = states_RL_DF.values[:, 100 * 2 + 1]

    states_act0_noRL = states_act0_noRL_DF.values[:, 100 * 4 + 1:]
    states_act1_noRL = states_act1_noRL_DF.values[:, 100 * 4 + 1:]
    states_RL = states_RL_DF.values[:, 100 * 4 + 1:]
    actions_RL = actions_RL_DF.values[:, 1:]

    debris_at_pump = states_RL_DF.values[:, 100 * 0 + args.X_pump]
    m1_at_pump = states_RL_DF.values[:, 100 * 1 + args.X_pump]
    m2_at_pump = states_RL_DF.values[:, 100 * 2 + args.X_pump]
    temp_at_pump = states_RL_DF.values[:, 100 * 3 + args.X_pump]
    new_at_pump = states_RL_DF.values[:, 100 * 4 + args.X_pump]

    wound_radius_list_a0 = healing_time(states_act0_noRL)
    wound_radius_list_a1 = healing_time(states_act1_noRL)
    wound_radius_list_rl = healing_time(states_RL)

    fig = plt.figure(figsize=(12, 12), num=4)
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t[1:] / 3.0, actions_RL)
    ax.set_title('RL Policies')

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t / 3.0, wound_radius_list_a0, color='g', linestyle='--', label='No Treatment')
    ax.plot(t / 3.0, wound_radius_list_a1, color='r', linestyle='-', label='ODE Treatment')
    ax.plot(t / 3.0, wound_radius_list_rl, color='b', linestyle='-', label='RL Treatment')
    ax.plot(t_ode, wound_radius_list_a0, color='m', linestyle='--', label='Twice of No Treatment')
    ax.set_xlabel(r't, days')
    ax.set_ylabel(r'wound radius, mm')
    ax.set_title('Wound Radius')
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(states_act0_noRL_YY1, states_act0_noRL_YY2, label='No Treatment')
    ax.plot(states_act1_noRL_YY1, states_act1_noRL_YY2, label='ODE Treatment')
    ax.plot(states_RL_YY1, states_RL_YY2, label='RL Treatment')
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.legend()
    ax.set_title('Infomation Flow')

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t / 3.0, debris_at_pump, label='Hemostoisi')
    ax.plot(t / 3.0, m1_at_pump, label='Infalmation')
    ax.plot(t / 3.0, m2_at_pump, label='Proliferation')
    ax.plot(t / 3.0, temp_at_pump, label='Maturations')
    ax.set_xlabel('t, days')
    ax.set_ylabel('Values at Zero position')
    ax.legend()

    fig.tight_layout()
    plt.savefig(args.figs_dir + 'test_all_rl_{}_anum_{}_ep{}.pdf'.format(args.alg_rl, args.action_size, model_idx))
    plt.close()


def apply_noRL_treatment(args, action):
    args.check_opt = True
    env = SimpleEnv(args)
    state = env.reset()
    state_buf, action_buf = [], []
    k = 0
    state_buf.append(state)
    for k in range(600):
        next_state, reward, done, info = env.step(action)
        print('Time: {} Action: {} New Tissue: {:.2f}'.format(k + 1, action, info[-1]))
        state = next_state
        state_buf.append(state)
    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    state_buf_df.to_csv(args.data_dir + 'state_a{}_info.csv'.format(action))

    return env.t_span[k + 1] / 3.0


def apply_RL_treatment(args, model_idx, agent, env):
    args.check_opt = False

    agent.model.load_state_dict(torch.load(args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, model_idx)))
    state = env.reset()
    state_buf, action_buf = [], []
    state_buf.append(state)
    k = 0
    for k in range(600):
        action = agent.act(state, 0)
        next_state, reward, done, info = env.step(action)
        state = next_state
        state_buf.append(state)
        action_buf.append(action)
        print('Algorithm: {} Episode: {} Time: {} Action: {:.2f} New Tissue: {:.2f}'.format(args.alg_rl, model_idx, k + 1, action, info[-1]))

    state_buf = np.array(state_buf)
    state_buf_df = pd.DataFrame(state_buf)
    state_buf_df.to_csv(args.data_dir + 'state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    action_buf = np.array(action_buf)
    action_buf_df = pd.DataFrame(action_buf)
    action_buf_df.to_csv(args.data_dir + 'action_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    return env.t_span[k + 1] / 3.0


def dqn_trainer(env, args, writer=None):
    agent = Agent_DQN(env, args)

    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    eps = args.eps_start
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        t = score = 0
        for t in range(args.max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            print('Episode: {}/{} Step: {}/{}\tAction: {:.1f} NewTissue: {:.2f}'.format(
                i_episode, args.n_episodes, t + 1, args.max_t, action, info[-1]), flush=True)
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
        t = score = 0
        for t in range(args.max_t):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            print('Episode: {}/{} Step: {}/{}\tAction: {:.2f} NewTissue: {:.2f}'.format(
                i_episode, args.n_episodes, t + 1, args.max_t, action, info[-1]), flush=True)
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
            writer.add_scalar('DaysTaken', env.t_span[t + 1] / args.Tc, i_episode)
            writer.add_scalar('AverageReward', np.mean(scores_window), i_episode)
        print('Algorithm: {} Episode: {}/{}\tAverage Score: {:.2f}'.format(
            args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window)), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
    return scores


def ppo_trainer(env, args, writer=None):
    agent = Agent_PPO(env, args)

    scores = []                         # List containing scores from each episode
    scores_window = deque(maxlen=5)   # last 100 scores
    for i_episode in range(1, args.n_episodes + 1):
        state = env.reset()
        t = score = 0
        for t in range(args.max_t):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # print('Episode: {}/{} Step: {}/{}\tAction: {} NewTissue: {:.2f}'.format(
            #     i_episode, args.n_episodes, t + 1, args.max_t, action, info[-1, 0]), flush=True)
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
        print('Algorithm: {} Episode: {}/{}\tAverage Score: {:.2f}'.format(
            args.alg_rl, i_episode, args.n_episodes, np.mean(scores_window)), flush=True)
        sys.stdout.flush()
        if i_episode % 5 == 0:
            torch.save(agent.model.state_dict(),
                       args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(env.action_size, i_episode))
    return scores


def a3c_trainer(env, args, writer=None):
    # Writer will output to ./runs/ directory by default
    # writer = SummaryWriter()
    model_dir = './res_sim/models_a3c/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    agent = Agent_A3C(env, args)
    agent.step()
    return agent


def trainer(args):

    args.model_dir = './res_sim/models/models_{}/'.format(args.alg_rl)
    args.data_dir = './res_sim/data/data_{}/'.format(args.alg_rl)
    args.figs_dir = './res_sim/figs/figs_{}/'.format(args.alg_rl)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    env = SimpleEnv(args)
    print('Algorithm: {} CheckOpt: {} ActionSize: {} UseGPU: {}'.format(
        args.alg_rl, args.check_opt, env.action_size, args.gpu))

    writer = SummaryWriter()
    # writer = None

    if args.alg_rl == 'dqn':
        scores = dqn_trainer(env, args, writer)
    elif args.alg_rl == 'a2c':
        scores = a2c_trainer(env, args, writer)
    elif args.alg_rl == 'a3c':
        scores = a3c_trainer(env, args, writer)
    elif args.alg_rl == 'ppo':
        scores = ppo_trainer(env, args, writer)
    else:
        assert False, 'Please specify RL algorithm!!!'
    return scores


def tester(args):
    args.model_dir = './res_sim/models/models_{}/'.format(args.alg_rl)
    args.data_dir = './res_sim/data/data_{}/'.format(args.alg_rl)
    args.figs_dir = './res_sim/figs/figs_{}/'.format(args.alg_rl)

    dirs = [args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    if not os.path.exists(args.data_dir + 'state_a{}_info.csv'.format(0)):
        apply_noRL_treatment(args, action=0)
    if not os.path.exists(args.data_dir + 'state_a{}_info.csv'.format(1)):
        apply_noRL_treatment(args, action=1)

    env = SimpleEnv(args)

    if args.alg_rl == 'dqn':
        agent = Agent_DQN(env, args)
    elif args.alg_rl == 'a2c':
        agent = Agent_A2C(env, args)
    elif args.alg_rl == 'a3c':
        agent = Agent_A3C(env, args)
    elif args.alg_rl == 'ppo':
        agent = Agent_PPO(env, args)
    else:
        assert False, 'Please specify RL algorithm!!!'

    model_visited = []
    model_idx = 5
    stop_cnt = 0
    while model_idx <= args.n_episodes:
        if stop_cnt > 10:
            break
        if model_idx not in model_visited:
            if os.path.isfile(args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(args.action_size, model_idx)):
                model_visited.append(model_idx)
                apply_RL_treatment(args, model_idx, agent, env)
                plotsRes(args, model_idx)
                model_idx += 5
                stop_cnt = 0
            else:
                print('Stop Cnt{}\tWaiting for {} model_anum_{}_ep_{}.pth'.format(stop_cnt, args.alg_rl, args.action_size, model_idx))
                time.sleep(700)
                stop_cnt += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = GetParameters()
    args.is_train = True
    if args.is_train:
        trainer(args)
    else:
        tester(args)