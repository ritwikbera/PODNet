from argparse import ArgumentParser
from types import SimpleNamespace
import os
import shutil
import numpy as np 
import json
import torch
from torch import Tensor, nn, optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from models import *
from config import *
from losses import *
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='robotarium', help='Enter minigrid, robotarium or circleworld')
parser.add_argument('--encoder_type', type=str, default='recurrent', help='Enter recurrent, attentive, or MLP')
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--launch_tb', type=bool, default=False)
parser.add_argument('--use_json', type=bool, default=False)
parser.add_argument('--json_addr', type=str, default=None)

args = parser.parse_args()

if args.use_json:
    args = vars(args)

    # if json address is not specified, use default
    if args['json_addr'] is None:
        json_data = json.load(open('sample_params.json', 'r'))
    else:
        json_data = json.load(open(args['json_addr'], 'r'))

    for key in json_data:
        args[key] = json_data[key]
    print('Params {}'.format(args))
    args = SimpleNamespace(**args)

torch.manual_seed(100)
#clean start
try:
    shutil.rmtree(args.log_dir)
except Exception as e:
    print('No {} directory present'.format(args.log_dir))

device = 'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu'

conf = config(args.dataset)

use_discrete = True if args.dataset=='minigrid' else False
PAD_TOKEN = -99
tau = 5.0
tau_min = 0.1
ANNEAL_RATE = 0.003

my_dataset = RoboDataset(
    PAD_TOKEN=PAD_TOKEN, 
    MAX_LENGTH=conf.MAX_LENGTH, 
    root_dir='data/'+args.dataset+'/')

dataloader = DataLoader(my_dataset, batch_size=conf.batch_size,
                    shuffle=True, num_workers=1)

model = PODNet(
    state_dim=conf.state_dim,
    action_dim=conf.action_dim,
    latent_dim=conf.latent_dim,
    categorical_dim=conf.categorical_dim,
    encoder_type=args.encoder_type,
    use_discrete=use_discrete,
    device=device)

MAX_LENGTH = conf.MAX_LENGTH
SEGMENT_SIZE = conf.SEGMENT_SIZE

optimizer = optim.Adam(model.parameters(), lr=args.lr)
writer = create_summary_writer(model, dataloader, args.log_dir+'/tensorboard')

model.to(device)
model.train()

def train_step(engine, batch):
    
    states, next_states, actions = batch
    model.reset(states.size(0)) #reset hidden states/option label for each new trajectory batch
    L_ODC, L_BC, L_TS, L_KL = 0,0,0,0

    # send data to device
    states = states.to(device)
    next_states = next_states.to(device)
    actions = actions.to(device)

    for i in range(int(MAX_LENGTH/SEGMENT_SIZE)):
        
        seg_start = i*SEGMENT_SIZE
        seg_end = (i+1)*SEGMENT_SIZE

        cur_state_segment = states[:,seg_start:seg_end]
        next_state_segment = next_states[:,seg_start:seg_end]
        action_segment = actions[:,seg_start:seg_end]

        mask = (cur_state_segment!=Tensor([PAD_TOKEN, PAD_TOKEN]).to(device)).type(torch.FloatTensor)[:,:,0]

        empty_segment = torch.ones(cur_state_segment.size())*PAD_TOKEN
        empty_segment = empty_segment.to(device)

        if torch.all(torch.eq(cur_state_segment,empty_segment)):
            break

        optimizer.zero_grad()

        action_pred, next_state_pred, c_t = model(cur_state_segment, tau)
        
        L_ODC += DynamicsLoss(next_state_segment, next_state_pred, PAD_TOKEN, device)
        L_BC += BCLoss(action_segment, action_pred, PAD_TOKEN, device, use_discrete)
        L_KL += args.beta*KLDLoss(c_t, mask, conf.categorical_dim, device)
        L_TS += args.alpha*TSLoss(c_t, mask, device)
    
    loss = L_ODC + L_BC + L_TS + L_KL
    loss.backward()
    optimizer.step()

    return L_ODC, L_BC, L_TS, L_KL, loss

trainer = Engine(train_step)

RunningAverage(output_transform=lambda x: x[-1].item()).attach(trainer, 'smooth loss')

training_saver = ModelCheckpoint(args.log_dir+'/checkpoints', filename_prefix="checkpoint", save_interval=1, n_saved=1, save_as_state_dict=True, create_dir=True)
to_save = {"model": model, "optimizer": optimizer} 

scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1e-2, end_value=1e-1, cycle_size=60)

trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, to_save) 
#trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_1)

@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    print('Running Loss {:.2f}'.format(engine.state.metrics['smooth loss']))
    
@trainer.on(Events.ITERATION_COMPLETED)
def tb_log(engine):
    writer.add_scalar("Dynamics Loss", engine.state.output[0].item(), engine.state.iteration)
    writer.add_scalar("Behavior Cloning Loss", engine.state.output[1].item(), engine.state.iteration)
    #writer.add_scalar("Temporal Smoothing Loss", engine.state.output[2].item(), engine.state.iteration)
    writer.add_scalar("KL Divergence Penalty", engine.state.output[3].item(), engine.state.iteration)
    writer.add_scalar("Total Loss", engine.state.output[-1].item(), engine.state.iteration)

@trainer.on(Events.EPOCH_COMPLETED)
def update_temp(engine):
    global tau 
    tau = np.maximum(tau * np.exp(-ANNEAL_RATE*engine.state.epoch), tau_min)

@trainer.on(Events.COMPLETED)
def cleanup(engine):
    if args.launch_tb:
        os.system('tensorboard --logdir={}/tensorboard'.format(args.log_dir))

trainer.run(dataloader, args.epochs)

def train():
    for i in range(args.epochs):
        for j, batch in enumerate(dataloader):
            L_ODC, L_BC, loss = train_step(batch)
            print('Loss {:.2f}'.format(loss))

#train()

writer.close()