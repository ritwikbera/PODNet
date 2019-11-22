''' eval.py
Loads model, generate new data, evaluate it, and save plotting data.

Currently hardcoded to work with CircleWorld data for debugging purposes.

Usage: python eval.py
'''
import sys, os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pickle

from netv2 import *
from utils import RoboDataset

torch.manual_seed(100)

# -----------------------------------------------
# PODNet parameters
STATE_DIM = 2
action_dim = 2
latent_dim = 1
categorical_dim = 2
NUM_HEADS = 2

PAD_TOKEN = 0
SEGMENT_SIZE = 512
TRAJ_BATCH_SIZE = 1
MAX_LENGTH = SEGMENT_SIZE*4  #set any integer multiple of SEGMENT_SIZE

# create PODNet and load trained model
model_addr = 'checkpoint.pth'
model_data = torch.load(model_addr)

model = PODNet(
    state_dim=STATE_DIM, 
    action_dim=action_dim, 
    latent_dim=latent_dim, 
    categorical_dim=categorical_dim,
    SEGMENT_SIZE=SEGMENT_SIZE,
    NUM_HEADS=NUM_HEADS)
model.load_state_dict(model_data['model_state_dict'])
model.eval()

# load evaluation data
robo_dataset = RoboDataset(PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE, root_dir='data/')
params = {'batch_size' : TRAJ_BATCH_SIZE, 'shuffle' : True, 'num_workers' : 4}
dataloader = DataLoader(robo_dataset, **params)

# generate predicted states for loaded dataset
for i, (states, true_next_states) in enumerate(dataloader):
    states = states.view(-1, SEGMENT_SIZE, STATE_DIM)
    next_state_pred, c_t = model(states)
    

#reshape for plotting
next_state_pred = next_state_pred.view(-1, STATE_DIM).detach().numpy()
true_next_states = true_next_states.view(-1, STATE_DIM).detach().numpy()
c_t = c_t.view(-1, latent_dim*categorical_dim).detach().numpy()

stop_index = 0
for i in range(len(true_next_states)):
    if np.array_equal(true_next_states[i],np.array([PAD_TOKEN, PAD_TOKEN])):
        stop_index = i
        break

time = np.arange(0,i,10)

# plot
os.makedirs('plots', exist_ok=True)
plt.figure()
plt.plot(true_next_states[:stop_index,0], 'b-', label='Truth')
plt.plot(true_next_states[:stop_index,1], 'r-')
plt.plot(next_state_pred[:stop_index,0], 'b--', label='Predicted')
plt.plot(next_state_pred[:stop_index,1], 'r--')
plt.xlabel('Time Steps')
plt.ylabel('Position')
plt.legend()
plt.tight_layout()
plt.savefig('plots/dynamics.png', dpi=300)
plt.show()

plt.figure()
c_t = c_t[0:stop_index:10]
plt.plot(time, np.argmax(c_t, axis=-1), 'ro')
plt.xlabel('Time Steps')
plt.ylabel('Option')
plt.legend()
plt.tight_layout()
plt.savefig('plots/options.png', dpi=300)
plt.show()