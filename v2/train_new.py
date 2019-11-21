import time 
import argparse
import math
import os
import glob
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np 

from netv2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dummy_test', default=False, type=bool)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--use_cuda', default=True, type=bool)
args = parser.parse_args()

torch.manual_seed(100)
device = 'cuda:0' if args.use_cuda and torch.cuda.is_available() else 'cpu'

STATE_DIM = 2
action_dim = 2
latent_dim = 1
categorical_dim = 3
NUM_HEADS = 2

PAD_TOKEN = 0
SEGMENT_SIZE = 512
TRAJ_BATCH_SIZE = 1
MAX_LENGTH = SEGMENT_SIZE*4  #set any integer multiple of SEGMENT_SIZE

def pad_trajectory(trajectory, PAD_TOKEN, MAX_LENGTH):
    padded_traj = Tensor([PAD_TOKEN]).repeat(MAX_LENGTH, trajectory.size(-1))
    padded_traj[:trajectory.size(0)] = trajectory
    return padded_traj

def segment_traj(trajectory, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE):
    trajectory = pad_trajectory(trajectory, PAD_TOKEN, MAX_LENGTH)
    traj_segments = torch.split(trajectory, SEGMENT_SIZE, dim=-2)
    traj_segment_batch = torch.stack(traj_segments).view(-1, SEGMENT_SIZE, trajectory.size(-1))
    return traj_segment_batch

def load_segment_stack(trajectory, PAD_TOKEN=0, MAX_LENGTH=2048, SEGMENT_SIZE=512):
    states = segment_traj(trajectory, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE)
    true_next_states = segment_traj(trajectory[1:], PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE)
    return states, true_next_states

class RoboDataset(Dataset):
    def __init__(self, root_dir='data/'):
        self.root_dir = root_dir

    def __len__(self):
        return len(glob.glob(self.root_dir+'*'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = glob.glob(self.root_dir+'*')[idx]
        print(file)
        traj_file = pd.read_csv(file)
        trajectory = Tensor(np.array([traj_file['x_t'], traj_file['y_t']]).T)
        return load_segment_stack(trajectory, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE)

robo_dataset = RoboDataset(root_dir='data/')

params = {'batch_size' : TRAJ_BATCH_SIZE, 'shuffle' : True, 'num_workers' : 4}
dataloader = DataLoader(robo_dataset, **params)

model = PODNet(
    state_dim=STATE_DIM, 
    action_dim=action_dim, 
    latent_dim=latent_dim, 
    categorical_dim=categorical_dim,
    SEGMENT_SIZE=SEGMENT_SIZE,
    NUM_HEADS=NUM_HEADS,
    device=device).to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train_step(cur_state_segment, next_state_segment, device):
    optimizer.zero_grad()
    next_state_pred, c_t = model(cur_state_segment)
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
    loss = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean()
    
    # print('Loss shape (should be scalar) :{}'.format(loss.size()))
    # print(loss)

    loss.backward()
    optimizer.step()

    return loss

best_model, optimizer_state = model.state_dict(), optimizer.state_dict()

def save(filename, model_state = model.state_dict(), optimizer_state = optimizer.state_dict()):
    torch.save({'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state}, filename)

def train(device):
    smooth_loss = 0
    smooth_coeff = 0.5
    min_loss = 0

    print('On device: {}'.format(device))

    for epoch in range(args.epochs):
        loss = 0
        for i, (states, true_next_states) in enumerate(dataloader):
            
            states = states.view(-1, SEGMENT_SIZE, STATE_DIM).to(device)
            true_next_states = true_next_states.view(-1, SEGMENT_SIZE, STATE_DIM).to(device)
            #print('Index: {}'.format(i))
            #print('States size {}'.format(states.size()))
            #print('True_Next_States size {}'.format(true_next_states.size()))
            
            loss += train_step(states, true_next_states, device)

        if epoch == 0:
            smooth_loss = loss
            min_loss = loss
            best_model, optimizer_state = model.state_dict(), optimizer.state_dict()
        else :
            smooth_loss = smooth_coeff*smooth_loss + (1-smooth_coeff)*loss
            if loss < min_loss:
                best_model, optimizer_state = model.state_dict(), optimizer.state_dict()

        print('Epoch: {} Loss: {}'.format(epoch, loss))
        print('Epoch: {} Smoothed Loss: {}'.format(epoch, smooth_loss))

    print('Training complete, saving best model so far .....')
    save('checkpoint.pth', model_state=best_model, optimizer_state=optimizer_state)

try:
    train(device) 
except KeyboardInterrupt:
    print('Interrupted, saving best model so far.....')
    save('checkpoint.pth', model_state=best_model, optimizer_state=optimizer_state)
    
