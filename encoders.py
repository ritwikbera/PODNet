import torch
from torch import Tensor, nn
import torch.nn.functional as F 
from layers import *
import pdb

class Hook():
    def __init__(self, module, backward=False):
        self.backward = backward
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

        if not self.backward:
            print('Input Tensor to {} is: {} \n'.format(module.__class__.__name__, self.input))
        else:
            print('Backpropagated gradient to {} is {} \n'.format(module.__class__.__name__, self.output))
        
        pdb.set_trace()
    
    def close(self):
        self.hook.remove()

class OptionEncoder_MLP(nn.Module):
    def __init__(self, state_dim, latent_dim, categorical_dim, 
        use_dropout=True, mlp_hidden=32, device='cpu'):
        super(OptionEncoder_MLP, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.fc1 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, latent_dim*categorical_dim)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.device = device

        #normalize over the concatenated [s_t c_t] feature vector
        feature_dim = state_dim + latent_dim*categorical_dim
        self.layer_norm = nn.LayerNorm(feature_dim, elementwise_affine=True)

        # add hook to visualize inputs to layer
        # hook_ln = Hook(self.layer_norm)
        # hook_in = Hook(self.fc1)

        # check if option inference is receiving gradients
        # hook_out = Hook(self.fc3, backward=True)

    def init_states(self, batch_size):
        self.c_t = torch.eye(self.categorical_dim)[0].repeat(batch_size,1,self.latent_dim).to(self.device)

    def forward(self, states, tau):
        steps = states.size(1)        
        c_stored = self.c_t.detach()
        self.c_t = self.c_t.detach()
        #unroll feedforward network as well
        for i in range(steps):
            state = states[:,i]
            state = state.view(state.size(0),1,state.size(-1))
            z = torch.cat((state, self.c_t), -1)

            # z = self.layer_norm(z)
            h1 = self.relu(self.fc1(z))

            if self.use_dropout:
                h1 = self.dropout(h1)
            h2 = self.relu(self.fc2(h1))
            if self.use_dropout:
                h2 = self.dropout(h2)
            q = self.fc3(h2)

            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            self.c_t = F.gumbel_softmax(q_y, tau=tau, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

            c_stored = torch.cat((c_stored, self.c_t), dim = 1)

        return c_stored[:,1:]

class OptionEncoder_TCN(nn.Module):
    def __init__(self, state_dim, latent_dim, categorical_dim, \
        use_dropout=True, num_channels=[32,32], kernel_size=4, dropout=0.1, device='cpu'):
        super(OptionEncoder_TCN, self).__init__()
        if use_dropout:
            dropout = 0.1
        else:
            dropout = None
        output_size = latent_dim*categorical_dim
        input_size = state_dim

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def init_states(self, batch_size):
        pass

    def forward(self, states, tau):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(states.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)

        q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
        c_stored = F.gumbel_softmax(q_y, tau=tau, hard=not self.training)
        c_stored = c_stored.view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)
        
        return c_stored

class OptionEncoder_Attentive(nn.Module):
    def __init__(self, state_dim, latent_dim, categorical_dim, NUM_HEADS=2, use_dropout=False, device='cpu'):
        super(OptionEncoder_Attentive, self).__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.pos_enc = PositionalEncoder(device)
        self.mhatt = MultiHeadAttention(state_dim+1, heads=NUM_HEADS)
        self.ffn = FeedForward(state_dim+1, latent_dim*categorical_dim)
        self.device = device

    def forward(self, s_t, tau):
        a = torch.arange(s_t.size()[-2])
        mask = (a[None, :] <= a[:, None]).type(torch.FloatTensor).to(self.device)
        s_t = self.pos_enc(s_t)
        q = self.ffn(self.mhatt(s_t, s_t, s_t, mask))

        q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
        c_stored = F.gumbel_softmax(q_y, tau=tau, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

        return c_stored

    def init_states(self, batch_size):
        pass

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim, categorical_dim, use_dropout=True, mlp_hidden=32):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(in_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, out_dim)
        self.relu = nn.ReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.3)

        #normalize over the concatenated [s_t c_t] feature vector
        feature_dim = in_dim + latent_dim*categorical_dim
        self.layer_norm = nn.LayerNorm(feature_dim, elementwise_affine=True)

    def forward(self, s_t, c_t):
        z = torch.cat((s_t, c_t), -1)
        # z = self.layer_norm(z)
        h1 = self.relu(self.fc1(z))
        if self.use_dropout:
            h1 = self.dropout(h1)
        h2 = self.relu(self.fc2(h1))
        if self.use_dropout:
            h2 = self.dropout(h2)
        return self.fc3(h2)

class OptionEncoder_Recurrent(nn.Module):
    def __init__(self, input_size, latent_dim, categorical_dim, device='cpu', hidden_layer_size=32, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.latent_dim =latent_dim
        self.categorical_dim = categorical_dim
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size+latent_dim*categorical_dim,
         hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, latent_dim*categorical_dim)
        
        self.init_params()

    def init_params(self):
        self.linear.bias = nn.Parameter(torch.eye(self.categorical_dim)[0])

    def init_states(self, batch_size):
        num_layers, hidden_layer_size = self.num_layers, self.hidden_layer_size
        self.hidden_cell = (torch.zeros(num_layers,batch_size,hidden_layer_size).to(self.device),
                            torch.zeros(num_layers,batch_size,hidden_layer_size).to(self.device))
        
        self.c_t = torch.eye(self.categorical_dim)[0].repeat(batch_size,1,self.latent_dim).to(self.device)

    def forward(self, states, tau):
        steps = states.size(1)

        #detach after full unrolling instead of after each step
        self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())
        self.c_t = self.c_t.detach()
        c_stored = self.c_t
        
        for i in range(steps):
            state = states[:,i]
            state = state.view(state.size(0),1,state.size(-1))

            lstm_in = torch.cat((state, self.c_t), dim=-1)
            lstm_out, self.hidden_cell = self.lstm(lstm_in, self.hidden_cell)

            q = self.linear(lstm_out)

            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            self.c_t = F.gumbel_softmax(q_y, tau=tau, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

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