import torch
from torch import Tensor, nn
from models import *
from utils import RoboDataset
from torch.utils.data import DataLoader

PAD_TOKEN = -99
MAX_LENGTH = 512
dataset = 'circleworld'

my_dataset = RoboDataset(
    PAD_TOKEN=PAD_TOKEN, 
    MAX_LENGTH=MAX_LENGTH, 
    root_dir='data/'+dataset+'/')

dataloader = DataLoader(my_dataset, batch_size=1,
                    shuffle=True, num_workers=1)

#states = torch.rand(1,492,2)

states, _, actions = next(iter(dataloader))

model = PODNet(
    batch_size=states.size(0),
    state_dim=states.size(-1),
    action_dim=actions.size(-1),
    latent_dim=1,
    categorical_dim=2,
    encoder_type='recurrent',
    device='cpu')

filename = 'mylogs/checkpoints/checkpoint_model_100.pth'
model.load_state_dict(torch.load(filename))
model.eval()
print('Saved Model Loaded')

def hook_fn(m, i, o):
    print('Option Inference Logits {}'.format(o))
model.infer_option.linear.register_forward_hook(hook_fn)

out = model(states)
print(model.decode_next_state)