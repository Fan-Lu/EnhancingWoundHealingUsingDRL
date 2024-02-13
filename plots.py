import random

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from cfgs.config import GetParameters
from model_tester import plotsRes

def mesh_2d_mat(t, r, tdays, radius, vector):
    @np.vectorize
    def value(tval, rval):
        tidx = np.where(t == tval)[0][0]
        ridx = np.where(r == rval)[0][0]
        val = vector[tidx][ridx]
        return val

    valuefun = value(tdays, radius)
    return valuefun


def plots3D(vectors):
    t = np.linspace(0, 60, 601)
    r = np.linspace(0, 3, 100)
    tdays, radius = np.meshgrid(t, r)

    fig = plt.figure(figsize=(16, 12), num=5)
    names = ['Debris', 'M1', 'M2', 'Temp Tissue', 'New Tissue']

    for cnt in range(5):
        vector = vectors[cnt]
        valuefun = mesh_2d_mat(t, r, tdays, radius, vector)
        ax = fig.add_subplot(2, 3, cnt + 1, projection='3d')
        ax.plot_surface(tdays, radius, valuefun, cmap = cm.coolwarm,
                         linewidth = 0, antialiased = False)
        ax.set_title(r'{}'.format(names[cnt]))
        ax.set_xlabel(r'radius $r$, mm')
        ax.set_ylabel(r't, days')

    new_tissue = vectors[-1]

    wound_radius_list = []
    for t_idx in range(601):
        new_tissue_t_day = new_tissue[t_idx, :]
        rmin = 99
        for ridx in range(100):
            if new_tissue_t_day[ridx] >= 0.95:
                rmin = ridx
                break
        wound_radius = r[rmin]
        wound_radius_list.append(wound_radius)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t / 3.0, wound_radius_list)
    ax.set_xlabel(r't, days')
    ax.set_ylabel(r'wound radius, mm')

    fig.tight_layout()
    plt.savefig('./res/figs/alvars_3d.pdf')
    plt.close()


def healing_time(new_tissues):
    tdays = np.linspace(0, 60, 601)
    new_tissue_t_day = new_tissues[:, 0]
    th = tdays[-1] / 3.0
    for t_idx in range(601):
        if new_tissue_t_day[t_idx] >= 0.95:
            th = tdays[t_idx] / 3.0
            break
    return th

if __name__ == "__main__":
    # states_RL_DF = pd.read_csv(args.data_dir + 'state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))

    # a_2d = pd.read_csv('./res/data/a_2d.csv').values[:, 1:]
    # m1_2d = pd.read_csv('./res/data/m1_2d.csv').values[:, 1:]
    # m2_2d = pd.read_csv('./res/data/m2_2d.csv').values[:, 1:]
    # c_2d = pd.read_csv('./res/data/c_2d.csv').values[:, 1:]
    # n_2d = pd.read_csv('./res/data/n_2d.csv').values[:, 1:]
    #
    # plots3D((a_2d, m1_2d, m2_2d, c_2d, n_2d))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for r in range(1, 11):
    #     ridx = 10 * r - 1
    #     ax.plot(np.linspace(0, 60, 601), n_2d[:, ridx], label=r'$r={}$'.format(ridx))
    # h, l = ax.get_legend_handles_labels()
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_xlabel('t, days')
    # plt.ticklabel_format(useOffset=False)
    # plt.tight_layout()
    # plt.savefig('./res/figs/allnew.pdf', format='pdf')
    # plt.close()

    # args = GetParameters()

    # ht = []

    # eps = np.arange(5, 30, 5)
    # for model_idx in eps:
    #     states_RL = pd.read_csv('./res/data/state_rl_anum_{}_ep_{}.csv'.format(args.action_size, model_idx))
    #     states_RL = states_RL.values[:, (100 * 4 + 1):]
    #
    #     wound_radius_list_rl = healing_time(states_RL)
    #
    #     ht.append(wound_radius_list_rl)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    #
    # ax.plot(eps, ht)
    # ax.set_xlabel('Model Episode')
    # ax.set_ylabel('Healing Time, day')
    # plt.savefig('./res/figs/healingtime_a2c_anum_{}.pdf'.format(args.action_size), format='pdf')
    # plt.close()

    # plotsRes(args, 50)

    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    # import tmp102

    # Create figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []

    # Initialize communication with TMP102
    # tmp102.init()


    # This function is called periodically from FuncAnimation
    def animate(i, xs, ys):
        # Read temperature (Celsius) from TMP102
        temp_c = round(random.random())

        # Add x and y to lists
        xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        ys.append(temp_c)

        # Limit x and y lists to 20 items
        xs = xs[-20:]
        ys = ys[-20:]

        # Draw x and y lists
        ax.clear()
        ax.plot(xs, ys)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('TMP102 Temperature over Time')
        plt.ylabel('Temperature (deg C)')


    # Set up plot to call animate() function periodically

    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000, cache_frame_data=False)
    plt.show()