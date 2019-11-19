import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F 
import numpy as np 
import time 
import argparse

from netv2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--dummy_test', default=True, type=bool)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

torch.manual_seed(100)

PAD_TOKEN = 0
SEGMENT_SIZE = 128
BATCH_SIZE = 1
STATE_DIM = 2

action_dim = 2
latent_dim = 1
categorical_dim = 3

max_length = 6000
num_segments = max_length // SEGMENT_SIZE + 1

if args.dummy_test:
    num_segments = 1
    cur_state_segment = torch.ones(BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)
    next_state_segment = torch.ones(BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)

model = PODNet(
    state_dim=STATE_DIM, 
    action_dim=action_dim, 
    latent_dim=latent_dim, 
    categorical_dim=categorical_dim)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(cur_state_segment, next_state_segment):
    next_state_pred, c_t = model(cur_state_segment)
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor)
    loss = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean(0)
    
    print('Loss shape (should be scalar) :{}'.format(loss.size()))
    print(loss)

    loss.backward()
    optimizer.step()

for i in range(args.epochs):
    for j in range(num_segments):
        train(cur_state_segment, next_state_segment)

#if __name__ =='__main__':
