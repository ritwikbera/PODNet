import torch
from torch import Tensor, nn, optim
import numpy as np 
import csv

def to_one_hot(actions):
    assert actions.size(-1) == 1
    actions -= torch.min(actions)
    action_dim = int(torch.max(actions).item()+1)
    actions = \
    torch.zeros(*actions.size()[:-1], action_dim).scatter_(-1, actions.type(torch.LongTensor), 1)
    return actions

def generate_toy_data(num_symbols=3, num_segments=3, max_segment_len=10):
    """Generate toy data sample with repetition of symbols (EOS symbol: 0)."""
    seq = []
    symbols = np.random.choice( \
        # np.arange(1, num_symbols + 1), num_segments, replace=False)
        np.arange(0, num_symbols), num_segments, replace=False)

    for seg_id in range(num_segments):
        segment_len = np.random.choice(np.arange(5, max_segment_len))
        seq += [symbols[seg_id]] * segment_len

    seq = Tensor(seq).unsqueeze(-1)
    seq = to_one_hot(seq)
    actions = seq[1:]-seq[:-1]
    seq = seq[1:]

    seq_len = len(seq)
    pos = torch.arange(seq_len).type(torch.FloatTensor).unsqueeze(-1)/seq_len
    seq = torch.cat((seq, pos),-1)
    seq = torch.cat((seq, actions),-1)
    # seq += [0] #EOS token
    return seq.detach().numpy()

def gen_dataset(save_csv=True, save_csv_addr=None):
    if save_csv:
        if save_csv_addr == None:
            save_csv_addr = 'scalar_traj.csv'

        with open(save_csv_addr, mode='w') as csv_file:
            fieldnames = ['x1_t','x2_t','x3_t','t','a1','a2','a3']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    
        with open(save_csv_addr, mode='ab') as csv_file:
            np.savetxt( \
                csv_file, generate_toy_data(), delimiter=',')

if __name__=='__main__':
    for i in range(20):
    #    print(generate_toy_data().detach().numpy())
        gen_dataset(save_csv_addr='scalar_{}.csv'.format(i))
