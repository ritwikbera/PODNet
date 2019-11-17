''' plotting.py
Loads experiment data as pickle file and plot results.

Usage: python plotting.py <address to pickle file with results>
Example: python plotting.py CircleWorld_plot.pickle
'''
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.15)

import pickle

# load data
file_addr = sys.argv[1]
experiment_data = pickle.load(open(file_addr, "rb"))

# parse variables
env_name = experiment_data["env_name"]
exp_name = experiment_data["exp_name"]
traj_data = experiment_data["traj_data"]
traj_data_plot = experiment_data["traj_data_plot"]
true_segments_int = experiment_data["true_segments_int"]
c_t_plot = experiment_data["c_t_plot"]
loss_plot = experiment_data["loss_plot"]
action_dim = experiment_data["action_dim"]
state_dim = experiment_data["state_dim"]
categorical_dim = experiment_data["categorical_dim"]

# create saving directories
os.makedirs("results", exist_ok=True)
os.makedirs(f"results/{exp_name}", exist_ok=True)

# plot
if env_name != 'PerimeterDef':
    # general plotting function: plot all actions
    if action_dim > 2:
        n_plots_x = np.sqrt(action_dim)
        n_plots_y = n_plots_x
    else:
        n_plots_x = 2
        n_plots_y = 1
    plt.figure(figsize=[4*n_plots_x,2*n_plots_x])
    plt.suptitle('Evaluate Option-conditioned Policy')
    for i in range(action_dim):
        plt.subplot(n_plots_x,n_plots_y,i+1)
        p = plt.plot(traj_data[:,state_dim+i], '-', label='a{}'.format(i))
        plt.plot(traj_data_plot[:,int(state_dim/2)+i], '--', color=p[0].get_color(), label='a{}_pred'.format(i))
        plt.grid()

elif env_name == 'PerimeterDef':
    # spefic plotting function for the perimeter defense dataset
    n_robots = 8
    fig = plt.figure(figsize=[12,5])
    # plt.suptitle('Evaluate Option-conditioned Policy')
    for i in range(n_robots):
        ax = plt.subplot(2,4,i+1)
        if i < 6:
            ax.set_title('Defender {}'.format(i+1))
        else:
            ax.set_title('Intruder {}'.format(i+1-6))
        plt.plot(traj_data[:,state_dim+i], 'b-', label='vel_x')
        plt.plot(traj_data[:,state_dim+i+1], 'r-', label='vel_y')
        plt.plot(traj_data_plot[:,int(state_dim/2)+i], 'b--', label='vel_x_pred'.format(i))
        plt.plot(traj_data_plot[:,int(state_dim/2)+i+1], 'r--', label='vel_y_pred'.format(i))
        plt.ylim([-0.003, 0.003])
        plt.xlim([0, 750])
        # remove internal ticks
        if i == 0:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        elif i==5 or i==6 or i==7:
            ax.yaxis.set_major_formatter(plt.NullFormatter())
        elif i !=4 :
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.NullFormatter())
    # add common labels
    fig.text(0.5, 0.03, 'Time Steps', ha='center', va='center')
    fig.text(0.06, 0.5, 'Velocity (m/s)', ha='center', va='center', rotation='vertical')
plt.savefig(f'results/{exp_name}/{env_name}_policy.png', dpi=600)

# dynamics
if env_name != 'PerimeterDef':
    # general plotting function: plot all states
    if int(state_dim/2) > 2:
        n_plots_x = np.sqrt(int(state_dim/2))
        n_plots_y = n_plots_x
    else:
        n_plots_x = 2
        n_plots_y = 1
    plt.figure(figsize=[4*n_plots_x,2*n_plots_x])
    plt.suptitle('Evaluate Option-conditioned Dynamics')
    for i in range(int(state_dim/2)):
        plt.subplot(n_plots_x,n_plots_y,i+1)
        p = plt.plot(traj_data[:,i], '-', label='s{}'.format(i))
        plt.plot(traj_data_plot[:,i], '--', color=p[0].get_color(), label='s{}_pred'.format(i))
        plt.grid()

elif env_name == 'PerimeterDef':
    # spefic plotting function for the perimeter defense dataset
    n_robots = 8
    fig = plt.figure(figsize=[12,5])
    # plt.suptitle('Evaluate Option-conditioned Policy')
    for i in range(n_robots):
        ax = plt.subplot(2,4,i+1)
        if i < 6:
            ax.set_title('Defender {}'.format(i+1))
        else:
            ax.set_title('Intruder {}'.format(i+1-6))
        plt.plot(traj_data[:,i], 'b-', label='x')
        plt.plot(traj_data[:,i+1], 'r-', label='y')
        plt.plot(traj_data_plot[:,i], 'b--', label='x_pred'.format(i))
        plt.plot(traj_data_plot[:,i+1], 'r--', label='y_pred'.format(i))
        # plt.ylim([-1.5, 0.0])
        # plt.xlim([0, 750])
        # remove internal ticks
        if i == 0:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        elif i==5 or i==6 or i==7:
            pass
            # ax.yaxis.set_major_formatter(plt.NullFormatter())
        elif i !=4 :
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            # ax.yaxis.set_major_formatter(plt.NullFormatter())
    # add common labels
    fig.text(0.5, 0.03, 'Time Steps', ha='center', va='center')
    fig.text(0.06, 0.5, 'Position (m)', ha='center', va='center', rotation='vertical')

plt.savefig(f'results/{exp_name}/{env_name}_dynamics.png', dpi=600)

# inference
pred_segments = np.argmax(c_t_plot[:,:categorical_dim], axis=1)
if env_name == 'PerimeterDef':
    # move podnet output because there is not guarantee that podnet will find the
    # same label number as the ground truth

    # temp vars to store true labels
    pred_segments[pred_segments == 0] = 4
    pred_segments[pred_segments == 1] = 5
    pred_segments[pred_segments == 2] = 6
    pred_segments[pred_segments == 3] = 7

    # substitute
    pred_segments[pred_segments == 4] = 2
    pred_segments[pred_segments == 5] = 3
    pred_segments[pred_segments == 6] = 0
    pred_segments[pred_segments == 7] = 1

# plt.title('Evaluate Option Inference')
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(true_segments_int,'D',alpha=0.5,label='Ground Truth')
ax.plot(pred_segments, 'k.', label='PODNet')
ax.yaxis.set_major_locator(MultipleLocator(1))
plt.xlabel('Time Steps')
plt.legend(loc='center right')
# replace tick labels on y-axis by name of options
if env_name == 'PerimeterDef':
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels [1] = "Cyclic Pursuit"
    labels [2] = "Leader-Follower"
    labels [3] = "Formation: Circle"
    labels [4] = "Formation: Cone"
    ax.set_yticklabels(labels)
plt.tight_layout()
plt.savefig(f'results/{exp_name}/{env_name}_options.png', dpi=600)

# plot losses
plt.figure(figsize=[12,8])
plt.suptitle('Training Loss')
plt.subplot(321)
plt.plot(loss_plot[:,0], loss_plot[:,1], label='train_loss')
plt.legend()
plt.subplot(322)
plt.plot(loss_plot[:,0], loss_plot[:,2], label='L_BC_epoch')
plt.legend()
plt.subplot(323)
plt.plot(loss_plot[:,0], loss_plot[:,3], label='L_ODC_epoch')
plt.legend()
plt.subplot(324)
plt.plot(loss_plot[:,0], loss_plot[:,4], label='Reg_epoch')
plt.legend()
plt.subplot(325)
plt.plot(loss_plot[:,0], loss_plot[:,6], label='L_TSR_epoch')
plt.legend()
plt.subplot(326)
plt.plot(loss_plot[:,0], loss_plot[:,5], label='temp')
plt.legend()
plt.savefig(f'results/{exp_name}/{env_name}ning_loss.png')

plt.show()