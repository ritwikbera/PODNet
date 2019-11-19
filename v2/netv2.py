import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.autograd import Variable

from gumbel import *
from layers import *

class OptionEncoder(nn.Module):
    def __init__(self, state_dim, categorical_dim, latent_dim=1, use_dropout=False, mlp_hidden=32):
        super(OptionEncoder, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.mhatt = MultiHeadAttention(state_dim, latent_dim*categorical_dim)

    def forward(self, s_t):
        a = torch.arange(s_t.size()[1])
        mask = (a[None, :] <= a[:, None]).type(torch.FloatTensor)
        return self.mhatt(s_t, s_t, s_t, mask)
        
class OptionDynamics(nn.Module):
    def __init__(self, state_dim, categorical_dim, latent_dim=1, use_dropout=True, mlp_hidden=32):
        super(OptionDynamics, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.use_dropout = use_dropout

        self.fc1 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, state_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, s_t, c_t):
        z = torch.cat((s_t, c_t), -1)
        h1 = self.relu(self.fc1(z))
        if self.use_dropout:
            h1 = self.dropout(h1)
        h2 = self.relu(self.fc2(h1))
        if self.use_dropout:
            h2 = self.dropout(h2)
        return self.fc3(h2)

class PODNet(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, categorical_dim, temperature=0.1):
        super(PODNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temperature = temperature
        self.infer_option = OptionEncoder(state_dim, categorical_dim, latent_dim=latent_dim)
        self.decode_next_state = OptionDynamics(state_dim, categorical_dim, latent_dim=latent_dim)

    def forward(self, s_t):
        c_t_logits = self.infer_option(s_t)
        c_t_logits = c_t_logits.view(s_t.size()[0], s_t.size()[1], self.latent_dim, self.categorical_dim)
        c_t = gumbel_softmax(c_t_logits, self.latent_dim, self.categorical_dim, self.temperature)  
        next_state_pred = self.decode_next_state(s_t, c_t)
        return next_state_pred, c_t

if __name__=='__main__':
    SEGMENT_SIZE = 128
    BATCH_SIZE = 1
    state_dim = 2
    action_dim = 2
    latent_dim = 1
    categorical_dim = 3
    s_t = torch.randn(BATCH_SIZE, SEGMENT_SIZE, 2)
    model = PODNet(state_dim, action_dim, latent_dim, categorical_dim)
    next_state_pred, c_t = model(s_t)
    print('Next State size {}'.format(next_state_pred.size()))
    print('Option Label Size{}'.format(c_t.size()))


