import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

state_dim = 2
action_dim = 2
latent_dim = 2 #has a similar effect to multi-head attention
categorical_dim = 3 #number of options to be discovered 

#from Directed-InfoGAIL
temp = 5.0
temp_min = 0.1
ANNEAL_RATE = 0.003 
learning_rate = 1e-3
mlp_hidden = 64

epochs = 25
seed = 1   #required for random gumbel sampling
hard = False  

torch.manual_seed(seed)

#randomly generated trajectory data
traj_length = 100
traj_num = 1 #similar to minibatch size (how many traj to be simultaneously processed)
traj_data = torch.randn(traj_length, traj_num, state_dim + action_dim)

def gen_circle_traj(r_init, fin, plot_traj=False):
    state = []
    r = r_init

    for theta in np.arange(0, fin, 1):
        x = r*np.cos(theta*np.pi/180)
        y = r*np.sin(theta*np.pi/180)
        state.append([x, y])
    state = np.array(state)
    
    state = state.tolist()
    for theta in np.arange(fin, fin/2, -1):
        x = r*np.cos(theta*np.pi/180)
        y = r*np.sin(theta*np.pi/180)
        state.append([x, y])
       # r *= 0.99
    state = np.array(state)

    action = []
    for i in range(1, len(state)):
        action.append(state[i]-state[i-1])
    action.append([0, 0])
    action = np.array(action)

    # plot generated trajectories
    if plot_traj:
        plt.figure()
        plt.plot(state[:,0], label='s0')
        plt.plot(state[:,1], label='s1')
        plt.legend()

        plt.figure()
        plt.plot(action[:,0], 'o', label='a0')
        plt.plot(action[:,1], 'o', label='a1')
        plt.legend()

        plt.show()

    return torch.Tensor(np.expand_dims(np.hstack((state, action)), axis=1))

traj_data = gen_circle_traj(10,180,plot_traj=False)
traj_length = len(traj_data)

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

        #option inference layers
        self.fc1 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_hidden)        
        self.fc3 = nn.Linear(mlp_hidden, latent_dim * categorical_dim)

        #policy network layers
        self.fc4 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc5 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc6 = nn.Linear(mlp_hidden, action_dim)

        #option dynamics layers
        self.fc7 = nn.Linear(state_dim + latent_dim*categorical_dim, mlp_hidden)
        self.fc8 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc9 = nn.Linear(mlp_hidden, state_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)
        
    def decode_next_state(self, s_t, c_t):
        z = torch.cat((s_t, c_t), -1)
        h1 = self.relu(self.fc4(z))
        h2 = self.relu(self.fc5(h1))
        return self.fc6(h2)

    def decode_action(self, s_t, c_t):
        z = torch.cat((s_t, c_t), -1)
        h1 = self.relu(self.fc7(z))
        h2 = self.relu(self.fc8(h1))
        return self.fc9(h2)

    def forward(self, s_t, c_prev, temp, hard):
        x = torch.cat((s_t, c_prev), -1)
        q = self.encode(x.view(-1, state_dim + latent_dim*categorical_dim))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        c_t = gumbel_softmax(q_y, temp, hard)

        action_pred = self.decode_action(s_t, c_t)
        next_state_pred = self.decode_next_state(s_t, c_t)

        return action_pred, next_state_pred, c_t

# create PODNet
model = PODNet(temp)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_function(next_state_pred, true_next_state, action_pred, true_action, c_t, print_loss=False):
    #loss-function weights
    Lambda_1 = 0  
    Lambda_2 = 1
    
    beta = 0.0

    loss_fn = torch.nn.MSELoss(reduction='sum')
    L_BC = Lambda_2 * loss_fn(action_pred, true_action)
    L_ODC = Lambda_1 * loss_fn(next_state_pred, true_next_state)
    MSE = L_ODC + L_BC

    qy = c_t
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    #if print_loss:
    #    print('L_ODC: {} L_BC: {} Reg: {}'.format(L_ODC, L_BC, beta*KLD))
    
    return L_BC, L_ODC, beta*KLD
    
    #return MSE + beta*KLD


def train(epoch):
    model.train()
    train_loss = 0
    global temp, epochs, traj_length 
    #c_t_initial = torch.eye(latent_dim,option_dim)
    c_t_initial = torch.Tensor([[1,0,0],[1,0,0]])
    c_t_stored = c_t_initial.view(-1, 1, latent_dim*categorical_dim)
    c_t_stored = c_t_stored.repeat(traj_length+1, 1, 1)

    i=0

    L_BC_epoch, L_ODC_epoch, Reg_epoch = 0, 0, 0

    for data in traj_data:
        state, action = data[:,:state_dim], data[:,state_dim:]
        i += 1
        # print(i)
        # print('state: ')
        # print(state)
        # print(' action:')
        # print(action)

        optimizer.zero_grad()
        c_prev = c_t_stored[i-1]

        action_pred, next_state_pred, c_t = model(state,c_prev,temp,hard)

        #c_t_stored = torch.cat((c_t_stored, c_t.unsqueeze(0)),0)
        c_t_stored[i] = c_t
        #loss = loss_function(next_state_pred, state, action_pred, action, c_t)
        L_BC, L_ODC, Reg = loss_function(next_state_pred,state,action_pred,action,c_t)
        
        L_BC_epoch += L_BC.item()
        L_ODC_epoch += L_ODC.item()
        Reg_epoch += Reg.item()

        loss = L_BC + L_ODC + Reg
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()

    temp = np.maximum(temp * np.exp(-ANNEAL_RATE*epoch), temp_min)
    print('====> Epoch: {} Average loss: {:.4f} || Raw loss value: {}'.format(
        epoch, train_loss / len(traj_data), loss.item()))

    print('L_ODC: {} L_BC: {} Reg: {}'.format(L_ODC_epoch/i, L_BC_epoch/i, Reg_epoch/i))
    
    # plt.plot(epoch, L_ODC_epoch/i, 'ro')
    # plt.plot(epoch, L_BC_epoch/i, 'go')
    # plt.plot(epoch, Reg_epoch/i, 'bo')

    plt.plot(epoch, loss.item(), 'bo')

    print('L_ODC: {} L_BC: {} Reg: {}'.format(L_ODC_epoch/i, L_BC_epoch/i, Reg_epoch/i))
    
    plt.plot(epoch, L_ODC_epoch/i, 'r.')
    plt.plot(epoch, L_BC_epoch/i, 'g.')
    plt.plot(epoch, Reg_epoch/i, 'b.')

    if epoch == epochs:
        print(c_t_stored)

def run():
    plt.figure()
    plt.title('Training Loss')
    for epoch in range(1, epochs + 1):
        train(epoch)

    plt.savefig('train.png')
    plt.show()

if __name__ == '__main__':
    run()