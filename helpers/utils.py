import torch
from torch import Tensor 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import glob
import numpy as np
from config import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

def to_device(items, device='cpu'):
    new_items = []
    for item in items:
        if isinstance(item, Tensor):
            item = item.to(device)
        new_items.append(item)
    return new_items

def segment_is_empty(cur_state_segment, PAD_TOKEN):
    try:
        device = cur_state_segment.get_device()
    except Exception:
        device = 'cpu'
    if device == -1:
        device = 'cpu'
    
    empty_segment = (torch.ones(cur_state_segment.size())*PAD_TOKEN).to(device)

    if torch.all(torch.eq(cur_state_segment,empty_segment)):
        return True
    return False

def to_categorical(data):
    if isinstance(data, Tensor):
        data = data.detach().numpy()
    cat_data = np.argmax(data, axis=-1)
    return cat_data

def shift_states(curr_states, i):
    shifted = curr_states[0].repeat(i,1)
    shifted = torch.cat((shifted, curr_states[:-i]), dim=0)
    return shifted 

def to_one_hot(actions):
    assert actions.size(-1) == 1
    actions -= torch.min(actions)
    action_dim = int(torch.max(actions).item()+1)
    actions = \
    torch.zeros(*actions.size()[:-1], action_dim).scatter_(-1, actions.type(torch.LongTensor), 1)
    return actions

def create_summary_writer(model, dataloader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    # inputs =  next(iter(dataloader))
    # try:
    #     writer.add_graph(model, inputs[0])
    # except Exception as e:
    #     print("Failed to save model graph: {}".format(e))
    return writer

def pad_trajectory(trajectory, PAD_TOKEN, MAX_LENGTH):
    padded_traj = Tensor([PAD_TOKEN]).repeat(MAX_LENGTH, trajectory.size(-1))
    padded_traj[:trajectory.size(0)] = trajectory
    return padded_traj

class RoboDataset(Dataset):
    def __init__(self, dataset, encoder_type, stack_count, PAD_TOKEN, MAX_LENGTH):
        self.root_dir = 'data/'+dataset+'/'
        self.PAD_TOKEN = PAD_TOKEN
        self.MAX_LENGTH = MAX_LENGTH
        self.dataset = dataset
        self.encoder_type = encoder_type
        self.stack_count = stack_count

    def __len__(self):
        return len(glob.glob(self.root_dir+'*.csv'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = glob.glob(self.root_dir+'*.csv')[idx]
        traj = pd.read_csv(file)
        
        if self.dataset == 'minigrid':
            states = Tensor(np.array(traj.loc[:,'x_t':'a_1'])[:,:-1])
            actions = Tensor(np.array(traj.loc[:,'a_1':]))
            actions = to_one_hot(actions)
        
        elif self.dataset == 'scalar':
            states = Tensor(np.array(traj.loc[:,'x1_t':'a1'])[:,:-1])
            actions = Tensor(np.array(traj.loc[:,'a1':'a3']))

        elif self.dataset == 'circleworld':
            states = Tensor(np.array(traj.loc[:,'x_t':'y_t']))
            actions = Tensor(np.array(traj.loc[:,'a_x':'a_y']))

        elif self.dataset == 'robotarium':
            states = Tensor(np.array(traj.loc[:,'x_t':'y_t']))
            actions = Tensor(np.array(traj.loc[:,'a_x':'a_y']))
        
        curr_states = pad_trajectory(states, self.PAD_TOKEN, self.MAX_LENGTH)
        actions = pad_trajectory(actions, self.PAD_TOKEN, self.MAX_LENGTH)
        
        pred_horizon = 1 #prediction horizon for dynamics
        next_states = pad_trajectory(states[pred_horizon:], self.PAD_TOKEN, self.MAX_LENGTH)
        
        if self.dataset == 'scalar':
            next_states =  next_states[:,:-1]
            
        # concatenate current and previous state for MLP encoder
        if self.encoder_type == 'MLP':

            # stack_count includes current state
            curr_states_ = curr_states
            for i in range(1,self.stack_count):
                prev_states = shift_states(curr_states_, i)
                curr_states = torch.cat((curr_states, prev_states), dim=-1)

        return curr_states, next_states, actions

def data_feeder(args, PAD_TOKEN=-99, shuffle=True):
    conf = config(args.dataset)

    if args.encoder_type == 'MLP':
        stack_count = int(input('Enter number of frames to be stacked \n'))
    else:
        stack_count = None
            
    my_dataset = RoboDataset(
        dataset=args.dataset,
        encoder_type=args.encoder_type,
        stack_count=stack_count,
        PAD_TOKEN=PAD_TOKEN, 
        MAX_LENGTH=conf.MAX_LENGTH)

    dataloader = DataLoader(my_dataset, batch_size=conf.batch_size, \
                        shuffle=shuffle, num_workers=1)

    return dataloader


if __name__ == '__main__':
    dataloader = data_feeder('scalar', 'MLP')
    batch = next(iter(dataloader))

    print(batch[0].size())
    print(batch[1].size())
    print(batch[2].size())

    print(batch[0][0]) #first sample in first batch (states only)
    # print(batch[2])