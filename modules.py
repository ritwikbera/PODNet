import torch
from torch import Tensor, nn
import torch.nn.functional as F 
from layers import *

class OptionEncoder_MLP(nn.Module):
    def __init__(self, batch_size, state_dim, latent_dim, categorical_dim, 
        use_dropout=False, mlp_hidden=32, device='cpu'):
        super(OptionEncoder_MLP, self).__init__()
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.fc1 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, latent_dim*categorical_dim)
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        self.device = device

        self.init_states()

    def init_states(self):
        self.c_t = torch.eye(self.categorical_dim)[0].repeat(self.batch_size,1,self.latent_dim)

    def forward(self, states, temp=0.1):
        steps = states.size(1)
        c_stored = self.c_t
        for i in range(steps):
            state = states[:,i]
            state = state.view(state.size(0),1,state.size(-1))
    
            z = torch.cat((state, self.c_t.detach()), -1)
            h1 = self.relu(self.fc1(z))
            if self.use_dropout:
                h1 = self.dropout(h1)
            h2 = self.relu(self.fc2(h1))
            if self.use_dropout:
                h2 = self.dropout(h2)
            q = self.fc3(h2)

            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            self.c_t = F.gumbel_softmax(q_y, tau=temp, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

            c_stored = torch.cat((c_stored, self.c_t), dim = 1)

        return c_stored[:,1:]

class OptionEncoder_Attentive(nn.Module):
    def __init__(self, state_dim, latent_dim, categorical_dim, NUM_HEADS=2, use_dropout=False, device='cpu'):
        super(OptionEncoder_Attentive, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.pos_enc = PositionalEncoder()
        self.mhatt = MultiHeadAttention(state_dim+1, heads=NUM_HEADS)
        self.ffn = FeedForward(state_dim+1, latent_dim*categorical_dim)
        self.device = device

    def forward(self, s_t, temp=0.1):
        a = torch.arange(s_t.size()[-2])
        mask = (a[None, :] <= a[:, None]).type(torch.FloatTensor).to(self.device)
        s_t = self.pos_enc(s_t)
        q = self.ffn(self.mhatt(s_t, s_t, s_t, mask))

        q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
        c_stored = F.gumbel_softmax(q_y, tau=temp, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

        return c_stored

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, categorical_dim, use_dropout=True, mlp_hidden=32):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(in_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, out_dim)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
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

class OptionEncoder_Recurrent(nn.Module):
    def __init__(self, batch_size, input_size, latent_dim, categorical_dim, device='cpu', hidden_layer_size=32, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.batch_size = batch_size
        self.latent_dim =latent_dim
        self.categorical_dim = categorical_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size+latent_dim*categorical_dim,
         hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, latent_dim*categorical_dim)
        
        self.init_states()
        self.init_params()

    def init_params(self):
        self.linear.bias = nn.Parameter(self.c_t[0,0,:])

    def init_states(self):
        num_layers, batch_size, hidden_layer_size = \
        self.num_layers, self.batch_size, self.hidden_layer_size
        self.hidden_cell = (torch.zeros(num_layers,batch_size,hidden_layer_size),
                            torch.zeros(num_layers,batch_size,hidden_layer_size))
        
        self.c_t = torch.eye(self.categorical_dim)[0].repeat(self.batch_size,1,self.latent_dim)

    def forward(self, states, temp = 0.1):
        steps = states.size(1)
        c_stored = self.c_t
        for i in range(steps):
            state = states[:,i]
            state = state.view(state.size(0),1,state.size(-1))

            lstm_in = torch.cat((state, self.c_t.detach()), dim=-1)
            lstm_out, self.hidden_cell = self.lstm(lstm_in, 
                (self.hidden_cell[0].detach(), self.hidden_cell[1].detach()))
            
            q = self.linear(lstm_out)

            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            self.c_t = F.gumbel_softmax(q_y, tau=temp, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

            c_stored = torch.cat((c_stored, self.c_t), dim = 1)
        
        return c_stored[:,1:]  #return inferred options for the entire new segment

if __name__ == '__main__':
    states = torch.randn(4,10,2)
    latent_dim = 2
    categorical_dim = 3
    enc = OptionEncoder_Recurrent(states.size(0),states.size(-1), latent_dim, categorical_dim)
    enc2 = OptionEncoder_Attentive(states.size(-1), latent_dim, categorical_dim)
    enc3 = OptionEncoder_MLP(states.size(0),states.size(-1), latent_dim, categorical_dim)

    enc.init_states()
    enc.eval()
    options = enc(states)
    print('Options size R {}'.format(options.size()))
    options2 = enc2(states)
    print('Options size A {}'.format(options2.size())) 
    options3 = enc3(states)
    print('Options size M {}'.format(options3.size()))