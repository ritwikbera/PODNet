from argparse import ArgumentParser
import os
import torch
from torch import Tensor, nn, optim 
import torch.nn.functional as F
from utils import *
from models import *
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--PAD_TOKEN', type=int, default=0)
parser.add_argument('--MAX_LENGTH', type=int, default=2048)
parser.add_argument('--SEGMENT_SIZE', type=int, default=512)
parser.add_argument('--latent_dim', type=int, default=1)
parser.add_argument('--categorical_dim', type=int, default=2)
parser.add_argument('--state_dim', type=int, default=2)
parser.add_argument('--action_dim', type=int, default=2)

args = parser.parse_args()

device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

my_dataset = RoboDataset(args.PAD_TOKEN, args.MAX_LENGTH, args.SEGMENT_SIZE)

dataloader = DataLoader(my_dataset, batch_size=args.batch_size,
                    shuffle=True, num_workers=1)

model = PODNet(
    batch_size=args.batch_size,
    state_dim=args.state_dim,
    action_dim=args.action_dim,
    latent_dim=args.latent_dim,
    categorical_dim=args.categorical_dim,
    device=device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
writer = create_summary_writer(model, dataloader, args.log_dir)

model.to(device)
model.train()

def train_step(batch, device=device):
    
    states, next_states, actions = batch
    model.reset() #reset hidden states/option label for each new trajectory batch

    for i in range(args.MAX_LENGTH/args.SEGMENT_SIZE):
        
        seg_start = i*args.SEGMENT_SIZE
        seg_end = (i+1)*args.SEGMENT_SIZE

        cur_state_segment = states[seg_start:seg_end]
        next_state_segment = next_states[seg_start:seg_end]
        action_segment = actions[seg_start:seg_end]

        optimizer.zero_grad()

        action_pred, next_state_pred, c_t = model(cur_state_segment)
        
        mask1 = (next_state_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
        mask2 = (action_segment!=PAD_TOKEN).type(torch.FloatTensor).to(device)
       
        #sum MSE across dimensions, length and average across batch

        L_ODC = (((next_state_segment - next_state_pred)**2)*mask1).sum(-1).sum(-2).mean()
        L_BC = (((action_segment - action_pred)**2)*mask2).sum(-1).sum(-2).mean()

        loss = L_ODC + L_BC
        loss.backward()
        optimizer.step()

    return L_ODC.item(), L_BC.item(), loss.item()

def train():
    for i, batch in enumerate(dataloader):
        L-ODC, L_BC, loss = train_step(batch)

writer.close()