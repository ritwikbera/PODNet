from argparse import ArgumentParser
from types import SimpleNamespace
import os, pdb, sys, shutil, json, pickle
import numpy as np 
import torch
from torch import Tensor, nn, optim, autograd 
import torch.nn.functional as F
from helpers import *
from models import *
from config import *
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--dataset', type=str, default='circleworld', help='Enter minigrid, scalar, robotarium or circleworld')
parser.add_argument('--encoder_type', type=str, default='MLP', help='Enter recurrent, attentive, or MLP')
parser.add_argument('--latent', type=str, default='concrete', help='Enter concrete or gaussian')
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=1.0)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--use_cuda', type=bool, default=False)
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

args = vars(args)
if args['latent'] == 'concrete':
    latent_dim = 1
    categorical_dim = conf.categorical_dim
else:
    latent_dim = 2
    categorical_dim = 1

args['latent_dim'] = latent_dim
args['categorical_dim'] = categorical_dim
args = SimpleNamespace(**args)

os.makedirs(args.log_dir)
filename = args.log_dir+'/argsfile'
outfile = open(filename, 'wb+')
pickle.dump(args, outfile)
outfile.close()

dataloader = data_feeder(args, PAD_TOKEN=PAD_TOKEN)

outputs = next(iter(dataloader))

state_dim = outputs[0].size(-1)
next_state_dim = outputs[1].size(-1)
action_dim = outputs[2].size(-1)

model = PODNet(
    state_dim=state_dim,
    next_state_dim=next_state_dim,
    action_dim=action_dim,
    args=args,
    device=device)

model.apply(weight_init)

MAX_LENGTH = conf.MAX_LENGTH
SEGMENT_SIZE = conf.SEGMENT_SIZE

optimizer = optim.Adam(model.parameters(), lr=args.lr)
writer = create_summary_writer(model, dataloader, args.log_dir)

model.to(device)
model.train()

def train_step(engine, batch):
    
    states, next_states, actions = batch
    model.reset(states.size(0)) #reset hidden states/option label for each new trajectory batch
    L_ODC, L_BC, L_TS, L_KL = 0,0,0,0

    # send data to device
    states, next_states, actions = to_device([states, next_states, actions], device)
    weights = class_weights(actions)

    for i in range(int(MAX_LENGTH/SEGMENT_SIZE)):

        # pdb.set_trace()
        
        seg_start, seg_end = i*SEGMENT_SIZE, (i+1)*SEGMENT_SIZE

        cur_state_segment = states[:,seg_start:seg_end]
        next_state_segment = next_states[:,seg_start:seg_end]
        action_segment = actions[:,seg_start:seg_end]

        mask = (cur_state_segment!=Tensor([PAD_TOKEN]).to(device)).type(torch.FloatTensor)[:,:,0]

        if segment_is_empty(cur_state_segment, PAD_TOKEN):
            break

        optimizer.zero_grad()

        action_pred, next_state_pred, c_t, logits = model(cur_state_segment, tau)
        
        L_ODC += args.lambda1*DynamicsLoss(next_state_segment, next_state_pred, PAD_TOKEN, device)
        L_BC += args.lambda2*BCLoss(action_segment, action_pred, PAD_TOKEN, device, use_discrete, weights=weights)
        L_KL += args.beta*KLDLoss(logits, mask, args, device)
        # L_TS += args.alpha*TSLoss(c_t, mask, args, device)
    
    loss = L_ODC + L_BC + L_KL # + L_TS # enable/disable L_TS
    loss.backward()
    optimizer.step()

    return L_ODC, L_BC, L_TS, L_KL, loss

trainer = Engine(train_step)

RunningAverage(output_transform=lambda x: x[-1].item()).attach(trainer, 'smooth loss')

training_saver = ModelCheckpoint(args.log_dir+'/checkpoints', filename_prefix="checkpoint", save_interval=1, n_saved=1, save_as_state_dict=False, create_dir=True)
to_save = {"network": model, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()} 

scheduler_1 = LinearCyclicalScheduler(optimizer, "lr", start_value=1e-2, end_value=1e-1, cycle_size=60)

trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, to_save) 
#trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_1)

@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    print('Running Loss {:.2f}'.format(engine.state.metrics['smooth loss']))
    print('L_BC {:.2f}'.format(engine.state.output[1].item()))
    print('L_KL {:.2f}'.format(engine.state.output[-2].item()))
    
@trainer.on(Events.ITERATION_COMPLETED)
def tb_log(engine):
    writer.add_scalar("Dynamics Loss", engine.state.output[0].item(), engine.state.iteration)
    writer.add_scalar("Behavior Cloning Loss", engine.state.output[1].item(), engine.state.iteration)
    # writer.add_scalar("Temporal Smoothing Loss", engine.state.output[2].item(), engine.state.iteration)
    writer.add_scalar("KL Divergence Penalty", engine.state.output[3].item(), engine.state.iteration)
    writer.add_scalar("Total Loss", engine.state.output[-1].item(), engine.state.iteration)

@trainer.on(Events.EPOCH_COMPLETED)
def update_temp(engine):
    global tau 
    tau = np.maximum(tau * np.exp(-ANNEAL_RATE*engine.state.epoch), tau_min)

# check for NaN gradients
# with autograd.detect_anomaly():
#     trainer.run(dataloader, args.epochs)

trainer.run(dataloader, args.epochs)

writer.close()