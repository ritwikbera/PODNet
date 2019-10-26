import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

state_dim = 3
action_dim = 3
latent_dim = 1
categorical_dim = 2 

#from Directed-InfoGAIL
temp = 5.0
temp_min = 0.1
ANNEAL_RATE = 0.003 
learning_rate = 0.0003

epochs = 10
seed = 1
batch_size = 1
hard = False


mlp_in = state_dim + categorical_dim
mlp_hidden = mlp_in//2+1

mlp = torch.nn.Sequential(torch.nn.Linear(D_in, H),
        torch.nn.ReLU(), torch.nn.Linear(H, D_out),)

torch.manual_seed(seed)

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

class PODNet(nn.Module):
    def __init__(self, temp):
        super(PODNet, self).__init__()

        self.fc1 = nn.Linear(mlp_in, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, latent_dim * categorical_dim)

        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.relu(self.fc2(h1))
        
    def decode_next_state(self, z):

    def decode_action(self, z):
        

    def forward(self, x, temp, hard):
        q = self.encode(x.view(-1, 784))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        z = gumbel_softmax(q_y, temp, hard)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())

model = PODNet(temp)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False) / x.shape[0]

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for state_data in traj_data:
        optimizer.zero_grad()
        recon_batch, qy = model(state_data, temp, hard)
        loss = loss_function(recon_batch, state_data, qy)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        
    temp = np.maximum(temp * np.exp(-ANNEAL_RATE*epoch), temp_min)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(traj_data)))

def run():
    for epoch in range(1, epochs + 1):
        train(epoch)

if __name__ == '__main__':
    run()
