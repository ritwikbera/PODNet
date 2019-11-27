import torch
from torch import Tensor, nn
from models import *
from config import *
from utils import RoboDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

PAD_TOKEN = -99.0
dataset = 'circleworld'

conf = config(dataset)

my_dataset = RoboDataset(
    PAD_TOKEN=PAD_TOKEN, 
    MAX_LENGTH=conf.MAX_LENGTH, 
    root_dir='data/'+dataset+'/')

dataloader = DataLoader(my_dataset, batch_size=1,
                    shuffle=True, num_workers=1)

filename = 'mylogs/checkpoints/checkpoint_model_200.pth'


def load_model(filename, conf, enc_type='attentive'):
    model = PODNet(
        batch_size=conf.batch_size,
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

podnet = load_model(filename, conf, enc_type='recurrent')

#plot first trajectory in the acquired batch
def plot_podnet(batch, index_within_batch):
    batch, i = batch, index_within_batch

    iterator = iter(dataloader)

    for _ in range(batch+1):
        states, true_next_states, actions = next(iterator)

    action_pred, next_state_pred, c_t = podnet(states, tau=0.1)


    padded_state = np.repeat([PAD_TOKEN], states.size(-1))


    states = states[i].detach().numpy()
    true_next_states = true_next_states[i].detach().numpy()
    actions = actions[i].detach().numpy()
    next_state_pred = next_state_pred[i].detach().numpy()
    action_pred = action_pred[i].detach().numpy()    
    c_t = c_t[i].detach().numpy()


    stop_index = 0
    for index in range(len(true_next_states)):
        if np.array_equal(true_next_states[index], padded_state):
            stop_index = index
            break

    time = np.arange(0,stop_index,10)

    # plot
    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.plot(true_next_states[:stop_index,0], 'b-', label='Truth')
    plt.plot(true_next_states[:stop_index,1], 'r-')
    plt.plot(next_state_pred[:stop_index,0], 'b--', label='Predicted')
    plt.plot(next_state_pred[:stop_index,1], 'r--')
    plt.xlabel('Time Steps')
    plt.ylabel('Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/dynamics.png', dpi=300)
    plt.show()

    plt.figure()
    c_t = c_t[0:stop_index:10]
    plt.plot(time, np.argmax(c_t, axis=-1), 'ro')
    plt.xlabel('Time Steps')
    plt.ylabel('Option')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/options.png', dpi=300)
    plt.show()

plot_podnet(0,0)


# def hook_fn(m, i, o):
#     print('Option Inference Logits {}'.format(o))
# model.infer_option.linear.register_forward_hook(hook_fn)

# out = model(states)
# print(model.decode_next_state)