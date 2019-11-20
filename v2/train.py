import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F 
import numpy as np 
import time 
import argparse
import pandas as pd 
import glob
import math

from netv2 import *

path = 'data/*'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dummy_test', default=False, type=bool)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

torch.manual_seed(100)

PAD_TOKEN = 0
STATE_DIM = 2

action_dim = 2
latent_dim = 1
categorical_dim = 3

SEGMENT_SIZE = 512
TRAJ_BATCH_SIZE = 1
MAX_LENGTH = SEGMENT_SIZE*4  #set any integer multiple of SEGMENT_SIZE

num_trajectories = 1
num_batches = math.ceil(num_trajectories/TRAJ_BATCH_SIZE)


def load_trajectory_batch(path, traj_batch_index, traj_batch_size):
    
    if traj_batch_index == num_batches-1:
        files = glob.glob(path)[traj_batch_index*traj_batch_size:]
    else:
        files = glob.glob(path)[ \
            traj_batch_index*traj_batch_size:(traj_batch_index+1)*traj_batch_size]
    
    trajectories = []
    for file in files:
        traj_file = pd.read_csv(file)
        trajectory = np.array([traj_file['x_t'], traj_file['y_t']]).T
        trajectories.append(Tensor(trajectory))

    return trajectories

def load_segment_batch(i):

    traj_batch = load_trajectory_batch(
        path=path, 
        traj_batch_index=i,
        traj_batch_size=TRAJ_BATCH_SIZE)

    traj = Tensor([PAD_TOKEN]).repeat(TRAJ_BATCH_SIZE, MAX_LENGTH, STATE_DIM)
    traj_ns = traj
    
    for i, trajectory in enumerate(traj_batch):
        traj[i][:trajectory.size(0)] = trajectory
        traj_ns[i][:trajectory.size(0)-1] = trajectory[1:]   

    traj = torch.split(traj, SEGMENT_SIZE, dim=-2) #split along length dimension
    traj = torch.stack(traj).view(-1, SEGMENT_SIZE, STATE_DIM)
    
    traj_ns = torch.split(traj_ns, SEGMENT_SIZE, dim=-2) #split along length dimension
    traj_ns = torch.stack(traj_ns).view(-1, SEGMENT_SIZE, STATE_DIM)

    return traj, traj_ns

model = PODNet(
    state_dim=STATE_DIM, 
    action_dim=action_dim, 
    latent_dim=latent_dim, 
    categorical_dim=categorical_dim)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train_step(cur_state_segment, next_state_segment):
    next_state_pred, c_t = model(cur_state_segment)
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor)
    loss = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean(0)
    
    print('Loss shape (should be scalar) :{}'.format(loss.size()))
    print(loss)

    loss.backward()
    optimizer.step()

def train(device):

    global num_batches

    print('On device: {}'.format(device))

    if args.dummy_test:
        num_batches = 1
        SEGMENT_BATCH_SIZE = 1
        cur_state_segment = torch.ones(SEGMENT_BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)
        next_state_segment = torch.ones(SEGMENT_BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)

    for i in range(args.epochs):
        for j in range(num_batches):
            if not args.dummy_test:
                cur_state_segment, next_state_segment = load_segment_batch(j)
            train_step(cur_state_segment, next_state_segment)


train(torch.device('cpu'))