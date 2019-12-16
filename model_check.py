import torch
from torch import Tensor, nn
from models import *
from config import *
from utils import *
import matplotlib.pyplot as plt
import os, glob, pdb
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='circleworld')
parser.add_argument('--encoder_type', type=str, default='MLP')
parser.add_argument('--log_dir', type=str, default='mylogs')
parser.add_argument('--max_steps', type=int, default=None)
args = parser.parse_args()

PAD_TOKEN = -99.0
latent_dim = 1

conf = config(args.dataset)

dataloader = data_feeder(args.dataset, args.encoder_type, PAD_TOKEN=PAD_TOKEN, shuffle=False)

torch.manual_seed(100)

files = glob.glob(args.log_dir+'/checkpoints/*.pth')
# print(files)
# pdb.set_trace()
podnet = torch.load(files[0], map_location=torch.device('cpu'))
podnet.load_state_dict(torch.load(files[1], map_location=torch.device('cpu')))
podnet.eval()
print('Saved Model Loaded')

def compare_plot(true, pred, stop_index, plot_dir, name='Position'):
    plt.figure()
    plt.plot(true[:stop_index,0], 'b-', label='Truth')
    plt.plot(true[:stop_index,1], 'r-')
    plt.plot(pred[:stop_index,0], 'b--', label='Predicted')
    plt.plot(pred[:stop_index,1], 'r--')
    plt.xlabel('Time Steps')
    plt.ylabel(name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_dir+'/{}.png'.format(name), dpi=600)

#plot first trajectory in the acquired batch
def plot_podnet(index, max_steps):
    iterator = iter(dataloader)

    batch_size = next(iter(dataloader))[0].size(0)
    batch, i = index//batch_size, index%batch_size

    for _ in range(batch+1):
        states, true_next_states, actions = next(iterator)

    # print(states.size(), true_next_states.size(), actions.size())

    podnet.reset(batch_size)
    action_pred, next_state_pred, c_t = podnet(states, tau=0.1)

    padded_state = np.repeat([PAD_TOKEN], true_next_states.size(-1))

    states = states[i].detach().numpy()
    true_next_states = true_next_states[i].detach().numpy()
    actions = actions[i].detach().numpy()
    next_state_pred = next_state_pred[i].detach().numpy()
    action_pred = action_pred[i].detach().numpy()    
    c_t = c_t[i].detach().numpy()

    # check if want to plot all steps or just a given number
    if max_steps is None:
        stop_index = 0
        for index_ in range(len(true_next_states)):
            if np.array_equal(true_next_states[index_], padded_state):
                stop_index = index_
                break
    else:
        stop_index = max_steps
    print(stop_index)

    # plot parameters
    plot_interval=1

    # plot
    plot_dir = args.log_dir+'/plots_{}/'.format(index)
    os.makedirs(plot_dir, exist_ok=True)

    compare_plot(true_next_states, next_state_pred, stop_index=stop_index, plot_dir=plot_dir, name='Position')

    if args.dataset == 'minigrid':
        x = np.arange(0,6,1)
        action_names = ['turn left', 'turn right', 'move forward', 'pick up key', 'drop the key', 'open door']
        plt.figure()
        fig, ax = plt.subplots(1,1)
        plt.plot(np.argmax(actions[:stop_index], axis=-1), 'bs', alpha=1.0, label='Truth')
        plt.plot(np.argmax(action_pred[:stop_index], axis=-1), 'ro', alpha=0.5, label='Predicted')
        ax.set_yticks(x)
        ax.set_yticklabels(action_names)
    else:
        compare_plot(actions, action_pred, stop_index=stop_index, plot_dir=plot_dir, name='Actions')

    options = np.argmax(c_t, axis=-1)

    # print(len(true_next_states[0:stop_index]))
    # print(len(c_t))
    # print(len(options_))

    plt.figure()
    if args.dataset == 'minigrid':
        option_colors = ['r', 'g', 'b'] #one color each for goto key/door/goal
        waypoints = ['key', 'door', 'goal']
        for i in range(len(waypoints)):
            plt.plot(states[i,-(2*i+2)], states[i,-(2*i+1)],'k^')
            plt.text(states[i,-(2*i+2)], states[i,-(2*i+1)],'goal')

    elif args.dataset == 'circleworld':
        option_colors = ['r', 'b']  # 2 options in circlworld only

    try:
        for i in range(stop_index):
            plt.plot(states[i,0], states[i,1], option_colors[options[i]]+'o')
    except:
        plt.plot(states[:stop_index,0], states[:stop_index,1])
    
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.savefig(plot_dir+'/actual_trajectory.png', dpi=600)

if args.dataset == 'scalar':
    states, next_states, actions = next(iter(dataloader))
    action_pred, next_state_pred, c_t = podnet(states, tau=0.1)
    i = 0
    states_ = states[i].detach().numpy() #first trajectory
    
    stop_index = 30

    print('trajectory length: {}'.format(stop_index))

    print('States :')
    print(to_categorical(states[i,:stop_index,0:3]))
    print('True Next States: ')
    print(to_categorical(next_states[i,:stop_index,:]))
    print('Predicted Next States: ')
    print(to_categorical(next_state_pred[i,:stop_index,:]))

    print('True Actions :')
    print(to_categorical(actions[i,:stop_index,:]))
    print('Predicted Actions: ')
    print(to_categorical(action_pred[i,:stop_index,:]))

    print('Predicted Options: ')
    print(to_categorical(c_t[i,:stop_index,:]))
else:
    for i in range(10):
        plot_podnet(i, args.max_steps)
