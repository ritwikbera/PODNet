import torch
from torch import Tensor, nn
from models import *
from config import *
from utils import RoboDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='minigrid')
parser.add_argument('--encoder_type', type=str, default='recurrent')
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--filename', type=str, default='checkpoint_model_100.pth')
parser.add_argument('--max_steps', type=int, default=None)
args = parser.parse_args()

PAD_TOKEN = -99.0

conf = config(args.dataset)

my_dataset = RoboDataset(
    PAD_TOKEN=PAD_TOKEN, 
    MAX_LENGTH=conf.MAX_LENGTH, 
    root_dir='data/'+args.dataset+'/')

dataloader = DataLoader(my_dataset, batch_size=conf.batch_size,
                    shuffle=False, num_workers=1)

filename = args.log_dir+'/checkpoints/'+args.filename
torch.manual_seed(100)

def load_model(filename, conf, enc_type):
    model = PODNet(
        state_dim=conf.state_dim,
        action_dim=conf.action_dim,
        latent_dim=conf.latent_dim,
        categorical_dim=conf.categorical_dim,
        encoder_type=enc_type,
        device='cpu')

    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model.eval()
    print('Saved Model Loaded')

    return model

podnet = load_model(filename, conf, enc_type=args.encoder_type)

#plot first trajectory in the acquired batch
def plot_podnet(batch, index_within_batch, max_steps):
    batch, i = batch, index_within_batch
    iterator = iter(dataloader)

    for _ in range(batch+1):
        states, true_next_states, actions = next(iterator)

    podnet.reset(states.size(0))
    action_pred, next_state_pred, c_t = podnet(states, tau=0.1)

    padded_state = np.repeat([PAD_TOKEN], states.size(-1))

    states = states[i].detach().numpy()
    true_next_states = true_next_states[i].detach().numpy()
    actions = actions[i].detach().numpy()
    next_state_pred = next_state_pred[i].detach().numpy()
    action_pred = action_pred[i].detach().numpy()    
    c_t = c_t[i].detach().numpy()

    # check if want to plot all steps or just a given number
    if max_steps is None:
        stop_index = 0
        for index in range(len(true_next_states)):
            if np.array_equal(true_next_states[index], padded_state):
                stop_index = index
                break
    else:
        stop_index = max_steps

    # plot parameters
    plot_interval=1

    # plot
    os.makedirs(args.log_dir+'/plots', exist_ok=True)
    plt.figure()
    plt.plot(true_next_states[:stop_index,0], 'b-', label='Truth')
    plt.plot(true_next_states[:stop_index,1], 'r-')
    plt.plot(next_state_pred[:stop_index,0], 'b--', label='Predicted')
    plt.plot(next_state_pred[:stop_index,1], 'r--')
    plt.xlabel('Time Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(args.log_dir+'/plots/dynamics.png', dpi=600)
    #plt.show()

    if args.dataset == 'minigrid':
        x = np.arange(0,6,1)
        action_names = ['turn left', 'turn right', 'move forward', 'pick up key', 'drop the key', 'open door']
        plt.figure()
        fig, ax = plt.subplots(1,1)
        plt.plot(np.argmax(actions[:stop_index], axis=-1), 'bs', alpha=1.0, label='Truth')
        plt.plot(np.argmax(action_pred[:stop_index], axis=-1), 'ro', alpha=0.5, label='Predicted')
        ax.set_yticks(x)
        ax.set_yticklabels(action_names)
        plt.xlabel('Time Steps')
        plt.ylabel('Discrete Actions')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(args.log_dir+'/plots/actions.png', dpi=600)

    plt.figure()
    options = np.argmax(c_t, axis=-1)
    time = np.arange(0,stop_index,plot_interval)
    c_t = c_t[0:stop_index:plot_interval]
    options_ = np.argmax(c_t, axis=-1)
    plt.plot(time, options_, '.k', label='Predicted')
    print(options_)
    plt.xlabel('Time Steps')
    plt.ylabel('Option')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(args.log_dir+'/plots/options.png', dpi=600)
    #plt.show()

    print(len(true_next_states[0:stop_index]))
    print(len(c_t))
    print(len(options_))

    plt.figure()
    if args.dataset == 'minigrid':
        option_colors = ['r', 'g', 'b'] #one color each for goto key/door/goal
        for i in range(stop_index):
            plt.plot(states[i,0], states[i,1], option_colors[options[i]]+'o')
    else:
        plt.plot(states[:stop_index,0], states[:stop_index,1])
    plt.grid()
    plt.axis('equal')
    plt.savefig(args.log_dir+'/plots/actual_trajectory.png', dpi=600)

plot_podnet(0,2, args.max_steps)


# def hook_fn(m, i, o):
#     print('Option Inference Logits {}'.format(o))
# model.infer_option.linear.register_forward_hook(hook_fn)

# out = model(states)
# print(model.decode_next_state)