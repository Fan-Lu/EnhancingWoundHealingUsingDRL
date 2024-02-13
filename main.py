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

from trainer import *
from cfgs.config import GetParameters


def run_train(args):

    args.model_dir = '../../../ExpDataDARPA/res_rl4wound/models/models_{}/'.format(args.alg_rl)
    args.data_dir = '../../../ExpDataDARPA/res_rl4wound/data/data_{}/'.format(args.alg_rl)
    args.figs_dir = '../../../ExpDataDARPA/res_rl4wound/figs/figs_{}/'.format(args.alg_rl)

    # args.model_dir = './res_wound_rl/res_healnet/models/models_{}/'.format(args.alg_rl)
    # args.data_dir = './res_wound_rl/res_healnet/data/data_{}/'.format(args.alg_rl)
    # args.figs_dir = './res_wound_rl/res_healnet/figs/figs_{}/'.format(args.alg_rl)

    dirs = [args.model_dir, args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    env = WoundEnv(args)
    # env = WoundIonEnv(args)
    # env = SimpleEnv(args)
    # env = HealNetEnv(args)

    runs_dir = '_'.join(('_'.join(time.asctime().split(' '))).split(':')) + '_alg_' + 'ppo'
    runs_dir = args.model_dir + '../../runs/runs_{}/{}'.format(args.alg_rl, runs_dir)
    # runs_dir = args.model_dir + './runs_{}/{}'.format(args.alg_rl, runs_dir)
    os.makedirs(runs_dir)

    writer = SummaryWriter(log_dir=runs_dir)
    # writer = None

    if args.alg_rl == 'dqn':
        scores = dqn_trainer(env, args, writer)
    elif args.alg_rl == 'a2c':
        scores = a2c_trainer(env, args, writer)
    elif args.alg_rl == 'a3c':
        scores = a3c_trainer(env, args, writer)
    elif args.alg_rl == 'ppo':
        scores = ppo_trainer(env, args, writer)
    elif args.alg_rl == 'td3':
        scores = td3_trainer(env, args, writer)
    else:
        assert False, 'Please specify RL algorithm!!!'
    return scores


def run_test(args):

    # args.model_dir = '../../../res_wound_rl/res_healnet/models/models_{}/'.format(args.alg_rl)
    # args.data_dir = '../../../res_wound_rl/res_healnet/data/data_{}/'.format(args.alg_rl)
    # args.figs_dir = '../../../res_wound_rl/res_healnet/figs/figs_{}/'.format(args.alg_rl)
    
    args.model_dir = './res/models/models_{}_lux_0913/'.format(args.alg_rl)
    args.data_dir = './res/data/data_{}_lux_0913/'.format(args.alg_rl)
    args.figs_dir = './res/figs/figs_{}_lux_0913/'.format(args.alg_rl)

    dirs = [args.data_dir, args.figs_dir]
    for dirtmp in dirs:
        if not os.path.exists(dirtmp):
            os.makedirs(dirtmp)

    env = WoundEnv(args)

    if env.action_space.shape[0] < 2:
        filecheck = args.data_dir + 'state_info.csv'
    else:
        filecheck = args.data_dir + 'state_a_{}_{}_info.csv'.format(0, args.X_pump)

    if not os.path.exists(filecheck):
        if env.action_space.shape[0] < 2:
            act_test = 0.0
        else:
            act_test = np.array([0, args.X_pump])
        apply_noRL_treatment(args, action=act_test)

    # if not os.path.exists(args.data_dir + 'state_a{}_info.csv'.format(1)):
    #     args.check_opt = True
    #     apply_noRL_treatment(args, action=1)
    #     args.check_opt = False

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

    # model_idx = 5
    # plotsRes(args, model_idx)

    model_visited = []
    model_idx = 345
    stop_cnt = 0
    while model_idx <= args.n_episodes:
        if stop_cnt > 10:
            break
        if model_idx not in model_visited:
            if os.path.isfile(args.model_dir + 'checkpoint_anum_{}_ep_{}.pth'.format(
                    env.action_space.shape[0], model_idx)):
                model_visited.append(model_idx)
                apply_RL_treatment(args, model_idx, agent, env)
                # plotsRes(args, model_idx)
                model_idx += 5
                stop_cnt = 0
            else:
                print('Stop Cnt{}\tWaiting for {} checkpoint_anum_{}_ep_{}.pth'.format(
                    stop_cnt, args.alg_rl, env.action_space.shape[0], model_idx))
                time.sleep(700)
                stop_cnt += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = GetParameters()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # args.is_train = False
    if args.is_train:
        run_train(args)
    else:
        run_test(args)