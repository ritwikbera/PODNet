''' plotting.py
Loads experiment data as pickle file and plot results.

Usage: python plotting.py <address to pickle file with results>
Example: python plotting.py CircleWorld_plot.pickle
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import pickle

# load data
file_addr = sys.argv[1]
experiment_data = pickle.load(open(file_addr, "rb"))

# parse variables
traj_data = experiment_data["traj_data"]
traj_data_plot = experiment_data["traj_data_plot"]
true_segments_int = experiment_data["true_segments_int"]
c_t_plot = experiment_data["c_t_plot"]
loss_plot = experiment_data["loss_plot"]
action_dim = experiment_data["action_dim"]
state_dim = experiment_data["state_dim"]
categorical_dim = experiment_data["categorical_dim"]

# plot
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
plt.savefig('eval_policy.png')

# dynamics
if int(state_dim/2) > 2:
    n_plots = np.sqrt(int(state_dim/2))
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
plt.savefig('eval_dynamics.png')

# inference
# postprocess true labels
plt.figure()
plt.title('Evaluate Option Inference')
plt.plot(true_segments_int,'D',alpha=0.5,label='truth')
plt.plot(np.argmax(c_t_plot[:,:categorical_dim], axis=1), 'k.', label='pred')
plt.legend()
plt.savefig('eval_options.png')

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
plt.savefig('training_loss.png')

plt.show()