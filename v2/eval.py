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

# ---------------------------------------------------------------------------- 
# plot
os.makedirs('plots', exist_ok=True)

# dynamics
next_state_pred = next_state_pred.detach().numpy()
plt.figure()
plt.plot(true_next_states[0, 0,:,0], 'b-', label='Truth')
plt.plot(true_next_states[0, 0,:,1], 'r-')
plt.plot(next_state_pred[0,:,0], 'b--', label='Predicted')
plt.plot(next_state_pred[0,:,1], 'r--')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.tight_layout()
plt.savefig('plots/dynamics.png', dpi=300)

# options
c_t = c_t.detach().numpy()
plt.figure()
plt.plot(np.argmax(c_t[0,:,:], axis=1), 'k.', label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Option')
plt.legend()
plt.tight_layout()
plt.savefig('plots/options.png', dpi=300)

plt.show()