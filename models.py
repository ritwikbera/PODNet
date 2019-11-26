import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from modules import *

class PODNet(nn.Module):
    def __init__(self, batch_size, state_dim, action_dim, latent_dim, categorical_dim, 
        encoder_type='recurrent', use_discrete=False, device='cpu'):
        super(PODNet, self).__init__()

        if encoder_type == 'recurrent':
            self.infer_option = OptionEncoder_Recurrent(
                batch_size=batch_size,
                input_size=state_dim, 
                latent_dim=latent_dim, 
                categorical_dim=categorical_dim,
                device=device)

        elif encoder_type == 'attentive':
            self.infer_option = OptionEncoder_Attentive(
                state_dim=state_dim,
                latent_dim=latent_dim,
                categorical_dim=categorical_dim,
                device=device)
        
        self.decode_next_state = Decoder(
            in_dim=state_dim, 
            out_dim=state_dim,
            latent_dim=latent_dim, 
            categorical_dim=categorical_dim)

        self.decode_action = Decoder(
            in_dim=state_dim,
            out_dim=action_dim, 
            latent_dim=latent_dim, 
            categorical_dim=categorical_dim)

        self.device = device
        self.use_discrete = use_discrete

    def reset(self):
        self.infer_option.init_states()

    def forward(self, s_t):
        c_t = self.infer_option(s_t)
        next_state_pred = self.decode_next_state(s_t, c_t)
        action_pred = self.decode_action(s_t, c_t)

        if self.use_discrete:
            action_pred = F.softmax(action_pred, dim=-1)

        return action_pred, next_state_pred, c_t

if __name__ == '__main__':
    states = torch.randn(4,10,2)
    
    model = PODNet(
        batch_size=states.size(0),
        state_dim=states.size(-1),
        action_dim=states.size(-1),
        latent_dim=1,
        categorical_dim=2,
        encoder_type='recurrent',
        device='cpu')

    model.reset()

    outputs = model(states)

    print(outputs[0].size())
    print(outputs[1].size())
    print(outputs[2].size())