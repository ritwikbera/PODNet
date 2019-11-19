import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F 
import numpy as np 
import time 

from netv2 import *
from layers import *
from gumbel import * 

PAD_TOKEN = 0
SEGMENT_SIZE = 128
BATCH_SIZE = 1
STATE_DIM = 2

action_dim = 2
latent_dim = 1
categorical_dim = 3

cur_state_segment = torch.randn(BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)
next_state_segment = torch.randn(BATCH_SIZE, SEGMENT_SIZE, STATE_DIM)

model = PODNet(STATE_DIM, action_dim, latent_dim, categorical_dim)

def train(cur_state_segment, next_state_segment):
    next_state_pred, c_t = model(cur_state_segment)
    mask = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor)
    loss = (((next_state_segment - next_state_pred)**2)*mask).sum(-1).sum(-2).mean(0)
    
    print('Loss shape (should be scalar) :{}'.format(loss.size()))
    print(loss)
    #loss.backward()
    #optim.step()

train(cur_state_segment, next_state_segment)
#if __name__ =='__main__':
