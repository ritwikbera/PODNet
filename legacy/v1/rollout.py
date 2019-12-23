''' rollout.py
Loads model and rollouts PODNet's network based on initial state and option.

Usage: python rollout.py <model address> <data for initial conditions> <traj_length>
Example: python rollout.py results/sample_robot/PerimeterDef_trained.pt data/sample_robots.csv 250
         python rollout.py results/full_stacked_robot/PerimeterDef_trained.pt data/16_log.csv 25
'''
import sys, os
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

from circleworld import gen_circle_traj
from utils import to_categorical, normalize, denormalize
from podnet import PODNet

# -----------------------------------------------
# Random seeds
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

## parse arguments
# parse model to be loaded
model_addr = sys.argv[1]
# parse evaluation dataset to be loaded
try:
    eval_file = sys.argv[2]
except:
    eval_file = None
model_data = torch.load(model_addr)
env_name = model_data['env_name']
exp_name = model_data['exp_name']
temp_min = model_data['temp_min']
hard = model_data['hard']
use_recurrent = model_data['use_recurrent']
state_dim = model_data['state_dim']
action_dim = model_data['action_dim']
categorical_dim = model_data['categorical_dim']
latent_dim = model_data['latent_dim']
c_initial = model_data['c_initial']
loss_plot = model_data['loss_plot']

# # if available, use single GPU to train PODNet
# TODO: evaluation was not working with GPU
device = torch.device("cpu")
# if torch.cuda.is_available():  
#     device = torch.device("cuda:0")
# else:  
#     device = torch.device("cpu")

# # GPU not working for recurrent case
# if use_recurrent:
#     device = torch.device("cpu")

# define which option to rollout
c_to_rollouts = [
    torch.Tensor([[1,0,0,0]]),
    torch.Tensor([[0,1,0,0]]),
    torch.Tensor([[0,0,1,0]]),
    torch.Tensor([[0,0,0,1]])
]
for c_to_rollout in c_to_rollouts:

    # load evaluation data
    if env_name == 'CircleWorld':
        # generate normalized trajectory data
        traj_data, traj_data_mean, traj_data_std, true_segments_int = gen_circle_traj(
            r_init=1, n_segments=2, plot_traj=False, save_csv=False)
        traj_length = len(traj_data)

        # convert trajectory segments to pytorch format after one-hot encoding
        true_segments = torch.from_numpy(to_categorical(true_segments_int))

    elif env_name == 'PerimeterDef':
        # option legends
        option_legends = [
            "CYCLIC PURSUIT",
            "LEADER-FOLLOWER",
            "FORMATION - CIRCLE",
            "FORMATION - CONE",
        ]
        # load dataset
        if eval_file == None:
            eval_file_addr = 'data/3_log.csv'
        else:
            eval_file_adrr = eval_file
        dataset = np.genfromtxt(eval_file_adrr, delimiter=',')
        traj_data, true_segments_int = dataset[:,:state_dim+action_dim], dataset[:,-1]
        traj_length = traj_data.shape[0]

        # normalize states and actions and convert to pytorch format
        traj_data, traj_data_mean, traj_data_std = normalize(traj_data)
        traj_data = torch.Tensor(np.expand_dims(traj_data, axis=1))

        # convert trajectory segments to pytorch format after one-hot encoding
        true_segments = torch.from_numpy(to_categorical(true_segments_int))

    # send data to current device (CPU or GPU)
    traj_data = traj_data.to(device)
    true_segments = true_segments.to(device)
    c_initial = c_initial.to(device)

    # create PODNet and load trained model
    model = PODNet(state_dim,action_dim,latent_dim,categorical_dim,use_recurrent=use_recurrent)
    model.device = device
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    # initialize variables
    current_temp = temp_min
    i=0
    c_prev = c_to_rollout
    # use initial states as starting point
    traj_length = int(sys.argv[3])
    state = traj_data[0][:,:state_dim] #+ torch.Tensor(np.random.rand(32))

    # create arrays to store plotting data
    action_pred_plot = np.zeros((traj_length-1, action_dim))
    # the /2 terms accounts for only predicting the next state (not previous)
    next_state_pred_plot = np.zeros((traj_length-1, int(state_dim/2)))
    c_t_plot = np.zeros((traj_length-1, latent_dim*categorical_dim))

    # loops until traj_length-1 because we need to use the next state as true_next_state
    for k in range(traj_length-1):
        i += 1

        # predict next actions, states, and options
        action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)

        # overwrite predict c_t to follow fixed c_to_rollout
        c_t = c_to_rollout

        # store predictions for plotting after training
        action_pred_plot[i-1] = action_pred.detach().numpy()
        next_state_pred_plot[i-1] = next_state_pred.detach().numpy()
        c_t_plot[i-1] = c_t.detach().numpy()  

        # use predicted states as current ones (rollout with predicted data)
        # stack with previous states
        state = torch.cat([next_state_pred, state[:,:int(state_dim/2)]], dim=1)      
        c_prev = c_to_rollout

    # denormalize data for plotting
    traj_data = traj_data.numpy().squeeze()
    # ground truth
    traj_data = denormalize(traj_data, traj_data_mean, traj_data_std)
    # predicted
    traj_data_plot = np.hstack((next_state_pred_plot, action_pred_plot))
    traj_data_plot_mean = np.hstack(( traj_data_mean[:int(state_dim/2)], traj_data_mean[state_dim:] ))
    traj_data_plot_std = np.hstack((traj_data_std[:int(state_dim/2)], traj_data_std[state_dim:]))
    traj_data_plot = denormalize(traj_data_plot, traj_data_plot_mean, traj_data_plot_std)

    # save plotting data as dict (to be plotted later)
    experiment_data = {
        "env_name": env_name,
        "exp_name": exp_name,
        "traj_data": traj_data,
        "true_segments_int": true_segments_int,
        "traj_data_plot": traj_data_plot,
        "c_t_plot": c_t_plot,
        "loss_plot": loss_plot,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "categorical_dim": categorical_dim
    }
    pickle.dump(experiment_data, open(f"results/{exp_name}/{env_name}_rollout.pickle", "wb"))

    # plot
    if env_name == 'PerimeterDef':
        plt.figure()
        plt.ylabel('Position Y')
        plt.xlabel('Position X')
        n_robots = 6
        n_intruders = 2
        # plot initial position and trajectory of each robot
        for i in range(n_robots):
            if i == 0:
                # plot robot final position with label
                plt.plot(traj_data_plot[-1,0], traj_data_plot[-1,1], 'b^', label='Defender', markersize=10)
            # plot robot final position
            p = plt.plot(traj_data_plot[-1,0+i], traj_data_plot[-1,1+i], 'b^', markersize=10)
            # plot robot initial position
            plt.plot(traj_data_plot[0,0+i], traj_data_plot[0,1+i], 'bx', markersize=10)
            # plot trajectory
            plt.plot(traj_data_plot[:,0+i], traj_data_plot[:,1+i], 'b--', color=p[0].get_color(), alpha=0.5)
        # plot initial position and trajectory of each robot
        for i in range(n_intruders):
            if i == 0:
                # plot robot final position with label
                plt.plot(traj_data_plot[-1,n_robots+0], traj_data_plot[-1,n_robots+1], 'rs', label='Intruder', markersize=10)
                # plot option being followed
                c_to_rollout_label = option_legends[np.argmax(c_to_rollout.numpy())]
                plt.plot(traj_data_plot[-1,n_robots+0], traj_data_plot[-1,n_robots+1], '-k', label=c_to_rollout_label, markersize=10)
            # plot robot final position
            p = plt.plot(traj_data_plot[-1,n_robots+i], traj_data_plot[-1,n_robots+1+i], 'rs', markersize=10)
            # plot robot initial position
            plt.plot(traj_data_plot[0,n_robots+0+i], traj_data_plot[0,n_robots+1+i], 'rx', markersize=10)
            # plot trajectory
            plt.plot(traj_data_plot[:,n_robots+i], traj_data_plot[:,n_robots+1+i], 'r--', color=p[0].get_color(), alpha=0.5)
            
        # plt.xlim([-1.5,0.25])
        # plt.ylim([-1.5,0.25])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"results/{exp_name}/{env_name}_opt{np.argmax(c_to_rollout.numpy())}_rollout.png".format(k))

        # dynamics
        n_plots_x = 4
        n_plots_y = 2
        plt.figure(figsize=[4*n_plots_x,3*n_plots_x])
        plt.suptitle('Fixed option - State debugging')
        for i in range(n_robots+n_intruders):
            plt.subplot(n_plots_x,n_plots_y,i+1)
            if i < n_robots:
                label_id = f'R{i+1}'
            else:
                label_id = f'I{i-n_robots+1}'
            plt.plot(traj_data_plot[:,0+i], 'b-', label='{}_X'.format(label_id))
            plt.plot(traj_data_plot[:,1+i], 'r-', label='{}_Y'.format(label_id))
            plt.legend()
        plt.savefig(f"results/{exp_name}/{env_name}_opt{np.argmax(c_to_rollout.numpy())}_rollout_time.png".format(k))