import torch
from torch import Tensor 
from torch.utils.data import Dataset
import pandas as pd
import glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def create_summary_writer(model, dataloader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    inputs =  next(iter(dataloader))
    try:
        writer.add_graph(model, inputs[0])
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

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
    def __init__(self, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE, root_dir='data/'):
        self.root_dir = root_dir
        self.PAD_TOKEN = PAD_TOKEN
        self.MAX_LENGTH = MAX_LENGTH
        self.SEGMENT_SIZE = SEGMENT_SIZE

    def __len__(self):
        return len(glob.glob(self.root_dir+'*.csv'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = glob.glob(self.root_dir+'*.csv')[idx]
        traj_file = pd.read_csv(file)
        
        states = Tensor(np.array([traj_file['x_t'], traj_file['y_t']]).T)
        actions = Tensor(np.array([traj_file['a_x'], traj_file['a_y']]).T)
        next_states = states[1:]
        
        states = pad_trajectory(states, self.PAD_TOKEN, self.MAX_LENGTH)
        actions = pad_trajectory(actions, self.PAD_TOKEN, self.MAX_LENGTH)
        next_states = pad_trajectory(next_states, self.PAD_TOKEN, self.MAX_LENGTH)
        
        return states, next_states, actions

if __name__ == '__main__':
    from torch.utils.data import DataLoader 
    my_dataset = RoboDataset(PAD_TOKEN=0, MAX_LENGTH=2048, SEGMENT_SIZE=512)
    dataloader = DataLoader(my_dataset, batch_size=1,
                    shuffle=True, num_workers=1)
    batch = next(iter(dataloader))

    print(batch[0].size())
    print(batch[1].size())
    print(batch[2].size())