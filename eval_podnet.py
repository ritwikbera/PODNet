''' eval_podnet.py
Loads model, generate new data, evaluate it, and save plotting data.

Usage: python eval_podnet.py <model address>
Example: python eval_podnet.py results/test/CircleWorld_trained.pt
'''
import sys, os
import torch
import numpy as np
import pickle

from circleworld import gen_circle_traj
from utils import to_categorical, normalize, denormalize
from podnet import PODNet

# parse arguments
PLOT_RESULTS = True
model_addr = sys.argv[1]
model_data = torch.load(model_addr)
env_name = model_data['env_name']
exp_name = model_data['exp_name']
temp_min = model_data['temp_min']
hard = model_data['hard']
state_dim = model_data['state_dim']
action_dim = model_data['action_dim']
categorical_dim = model_data['categorical_dim']
latent_dim = model_data['latent_dim']
c_initial = model_data['c_initial']
loss_plot = model_data['loss_plot']

# load model
model = PODNet()
model.load_state_dict(model_data['model_state_dict'])
model.eval()


# load evaluation data
if env_name == 'CircleWorld':
    # generate normalized trajectory data
    traj_data, traj_data_mean, traj_data_std, true_segments_int = gen_circle_traj(
        r_init=1, n_segments=3, plot_traj=False, save_csv=False)
    traj_length = len(traj_data)

    # convert trajectory segments to pytorch format after one-hot encoding
    true_segments = torch.from_numpy(to_categorical(true_segments_int))

elif env_name == 'PerimeterDef':
    # load dataset
    dataset = np.genfromtxt('data/sample_robots.csv', delimiter=',')
    traj_data, true_segments_int = dataset[:,:state_dim+action_dim], dataset[:,-1]
    traj_length = traj_data.shape[0]

    # normalize states and actions and convert to pytorch format
    traj_data, traj_data_mean, traj_data_std = normalize(traj_data)
    traj_data = torch.Tensor(np.expand_dims(traj_data, axis=1))

    # convert trajectory segments to pytorch format after one-hot encoding
    true_segments = torch.from_numpy(to_categorical(true_segments_int))

# initialize variables
current_temp = temp_min
c_prev = c_initial
i=0

# create arrays to store plotting data
action_pred_plot = np.zeros((traj_length-1, action_dim))
# the /2 terms accounts for only predicting the next state (not previous)
next_state_pred_plot = np.zeros((traj_length-1, int(state_dim/2)))
c_t_plot = np.zeros((traj_length-1, latent_dim*categorical_dim))

# loops until traj_length-1 because we need to use the next state as true_next_state
for k in range(traj_length-1):
    data = traj_data[k]
    state, action = data[:,:state_dim], data[:,state_dim:]
    i += 1

    # predict next actions, states, and options
    action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)

    # store predictions for plotting after training
    action_pred_plot[i-1] = action_pred.detach().numpy()
    next_state_pred_plot[i-1] = next_state_pred.detach().numpy()
    c_t_plot[i-1] = c_t.detach().numpy()        
    c_prev = c_t

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
pickle.dump(experiment_data, open(f"results/{exp_name}/{env_name}_plot.pickle", "wb"))

# plot
if PLOT_RESULTS:
    os.system(f"python plotting.py results/{exp_name}/{env_name}_plot.pickle")