import torch
from torch import Tensor, nn
import torch.nn.functional as F 

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
    def __init__(self, batch_size, input_size, latent_dim=1, categorical_dim=2, device='cpu', hidden_layer_size=32, num_layers=2):
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
    
    def init_states(self):
        num_layers, batch_size, hidden_layer_size, latent_dim, categorical_dim = \
        self.num_layers, self.batch_size, self.hidden_layer_size, self.latent_dim, self.categorical_dim
        self.hidden_cell = (torch.zeros(num_layers,batch_size,hidden_layer_size),
                            torch.zeros(num_layers,batch_size,hidden_layer_size))
        
        self.c_t = torch.eye(categorical_dim)[0].repeat(batch_size,1,latent_dim)

    def forward(self, states, temp = 0.1):
        steps = states.size(1)
        c_stored = self.c_t
        for i in range(steps):
            state = states[:,i]

            state = state.view(state.size(0),1,state.size(-1))
            lstm_in = torch.cat((state, self.c_t.detach()), dim=-1)
            lstm_out, self.hidden_cell = self.lstm(lstm_in, 
                (self.hidden_cell[0].detach(), self.hidden_cell[1].detach()))
            
            q = self.linear(lstm_out.view(*state.size()[:-1], -1))

            q_y = q.view(*q.size()[:-1], self.latent_dim, self.categorical_dim)
            self.c_t = F.gumbel_softmax(q_y, tau=temp, hard=not self.training).view(*q_y.size()[:-2], self.latent_dim*self.categorical_dim)

            c_stored = torch.cat((c_stored, self.c_t), dim = 1)
        
        return c_stored[:,1:]  #return inferred options for the entire new segment

if __name__ == '__main__':
    states = torch.randn(4,10,2)
    enc = OptionEncoder_Recurrent(states.size(0),states.size(-1))
    enc.init_states()
    enc.eval()
    options = enc(states)
    print('Options size {}'.format(options.size())) #squeeze for length