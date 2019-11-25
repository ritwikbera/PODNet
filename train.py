from argparse import ArgumentParser
import os
import shutil
import torch
from torch import Tensor, nn, optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from models import *
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--use_cuda', type=bool, default=False)
parser.add_argument('--PAD_TOKEN', type=int, default=-99)
parser.add_argument('--MAX_LENGTH', type=int, default=2048)
parser.add_argument('--SEGMENT_SIZE', type=int, default=512)
parser.add_argument('--latent_dim', type=int, default=1)
parser.add_argument('--categorical_dim', type=int, default=2)
parser.add_argument('--state_dim', type=int, default=2)
parser.add_argument('--action_dim', type=int, default=2)
parser.add_argument('--launch_tb', type=bool, default=False)

args = parser.parse_args()

#clean start
try:
    shutil.rmtree(args.log_dir)
except Exception as e:
    print('No {} directory present'.format(args.log_dir))

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
writer = create_summary_writer(model, dataloader, args.log_dir+'/tensorboard')

model.to(device)
model.train()

def train_step(engine, batch):
    
    states, next_states, actions = batch
    model.reset() #reset hidden states/option label for each new trajectory batch
    L_ODC, L_BC, L_TS = 0,0,0

    for i in range(int(args.MAX_LENGTH/args.SEGMENT_SIZE)):
        
        seg_start = i*args.SEGMENT_SIZE
        seg_end = (i+1)*args.SEGMENT_SIZE

        cur_state_segment = states[:,seg_start:seg_end]
        next_state_segment = next_states[:,seg_start:seg_end]
        action_segment = actions[:,seg_start:seg_end]

        empty_segment = torch.ones(cur_state_segment.size())*args.PAD_TOKEN
        if torch.all(torch.eq(cur_state_segment,empty_segment)):
            break

        optimizer.zero_grad()

        action_pred, next_state_pred, c_t = model(cur_state_segment)
        
        mask1 = (next_state_segment!=args.PAD_TOKEN).type(torch.FloatTensor).to(device)
        mask2 = (action_segment!=args.PAD_TOKEN).type(torch.FloatTensor).to(device)
       
        #sum MSE across dimensions, length and average across batch

        L_ODC += (((next_state_segment - next_state_pred)**2)*mask1).sum(-1).sum(-2).mean()
        L_BC += (((action_segment - action_pred)**2)*mask2).sum(-1).sum(-2).mean()

        L_TS += - 0.0*((c_t[:,1:,:]*c_t[:,:-1,:])*mask2[:,1:,:]).sum(-1).sum(-2).mean()
    
    loss = L_ODC + L_BC + L_TS
    loss.backward()
    optimizer.step()

    return L_ODC, L_BC, L_TS, loss

trainer = Engine(train_step)

RunningAverage(output_transform=lambda x: x[-1].item()).attach(trainer, 'smooth loss')

training_saver = ModelCheckpoint(args.log_dir+'/checkpoints', filename_prefix="checkpoint", save_interval=1, n_saved=1, save_as_state_dict=True, create_dir=True)
to_save = {"model": model, "optimizer": optimizer} 
trainer.add_event_handler(Events.EPOCH_COMPLETED, training_saver, to_save) 

@trainer.on(Events.EPOCH_COMPLETED)
def print_loss(engine):
    print('Running Loss {:.2f}'.format(engine.state.metrics['smooth loss']))
    
@trainer.on(Events.ITERATION_COMPLETED)
def tb_log(engine):
    writer.add_scalar("Dynamics Loss", engine.state.output[0].item(), engine.state.iteration)
    writer.add_scalar("Behavior Cloning Loss", engine.state.output[1].item(), engine.state.iteration)
    writer.add_scalar("Temporal Smoothing Loss", engine.state.output[2].item(), engine.state.iteration)
    writer.add_scalar("Total Loss", engine.state.output[-1].item(), engine.state.iteration)

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