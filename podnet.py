import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# state_dim = 2 # (x_t, y_t) of circle
state_dim = 4 # (x_t, y_t, x_prev, y_prev) of circle
action_dim = 2
latent_dim = 1 #has a similar effect to multi-head attention
categorical_dim = 2 #number of options to be discovered 

#from Directed-InfoGAIL
temp = 5.0
temp_min = 0.1
ANNEAL_RATE = 0.003
learning_rate = 1e-4
mlp_hidden = 32

epochs = 50
seed = 1   #required for random gumbel sampling
np.random.seed(seed)
hard = False 

torch.manual_seed(seed)

def normalize(data):
    '''Substracts mean and divide by standard deviation and returns statistics.'''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    norm_data = (data-mean)/std
    return norm_data, mean, std

def denormalize(norm_data, mean, std):
    ''' Denormalize data based on given mean and standard deviation.'''
    data = norm_data*std + mean
    return data

def gen_circle_traj(r_init, n_segments, plot_traj=False):
    state = []
    r = r_init
    last_fin = 0
    direction = -1

    for j in range(n_segments):
        # generate size of segment
        fin = np.random.randint(low=30,high=150)
        # randomize direction
        direction *= -1
        # generate states
        for theta in np.arange(last_fin, fin, direction):
            x = r*np.cos(theta*np.pi/180)
            y = r*np.sin(theta*np.pi/180)
            state.append([x, y])
        last_fin += fin
    state = np.array(state)

    # concatenates next states to current ones so the array of states becomes
    # (x_t, y_t, x_prev, y_prev) of circle
    next_states = state[1:,:]
    prev_states = state[:-1,:]
    state = np.hstack((next_states, prev_states))

    action = []
    action.append([0,0])
    for i in range(1, len(state)):
        action.append(state[i,:2]-state[i-1,:2])
    action = np.array(action)
    # smooth out abrupt changes in states
    action = np.clip(action,-.0175,.0175)

    # check if have the same number of samples for states and actions
    print('{} samples.'.format(state.shape[0]))
    assert state.shape[0] == action.shape[0]

    # plot generated trajectories
    if plot_traj:
        plt.figure()
        plt.plot(state[:,0], label='s0')
        plt.plot(state[:,1], label='s1')
        plt.plot(state[:,2], label='s2')
        plt.plot(state[:,3], label='s3')
        plt.legend()

        plt.figure()
        plt.plot(action[:,0], 'o', label='a0')
        plt.plot(action[:,1], 'o', label='a1')
        plt.legend()

        plt.show()
    
    # normalize generated data
    state, state_mean, state_std = normalize(state)
    action, action_mean, action_std = normalize(action)

    return torch.Tensor(np.expand_dims(np.hstack((state, action)), axis=1))

# generate normalized trajectory data
traj_data = gen_circle_traj(r_init=1, n_segments=4, plot_traj=True)
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
        # the *2 terms accounts for c_prev
        self.fc4 = nn.Linear(state_dim + latent_dim*categorical_dim*2, mlp_hidden)
        self.fc5 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc6 = nn.Linear(mlp_hidden, action_dim)

        #option dynamics layers
        # the *2 terms accounts for c_prev
        # the /2 terms accounts for only predicting the next state (not previous)
        self.fc7 = nn.Linear(state_dim + latent_dim*categorical_dim*2, mlp_hidden)
        self.fc8 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc9 = nn.Linear(mlp_hidden, int(state_dim/2))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2)
        
    def decode_next_state(self, s_t, c_t, c_prev):
        z = torch.cat((s_t, c_t, c_prev), -1)
        h1 = self.relu(self.fc4(z))
        h2 = self.relu(self.fc5(h1))
        return self.fc6(h2)

    def decode_action(self, s_t, c_t, c_prev):
        z = torch.cat((s_t, c_t, c_prev), -1)
        h1 = self.relu(self.fc7(z))
        h2 = self.relu(self.fc8(h1))
        return self.fc9(h2)

    def forward(self, s_t, c_prev, temp, hard):
        x = torch.cat((s_t, c_prev), -1)
        q = self.encode(x.view(-1, state_dim + latent_dim*categorical_dim))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        c_t = gumbel_softmax(q_y, temp, hard)

        action_pred = self.decode_action(s_t, c_t, c_prev)
        next_state_pred = self.decode_next_state(s_t, c_t, c_prev)

        return action_pred, next_state_pred, c_t

# create PODNet
model = PODNet(temp)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

def loss_function(next_state_pred, true_next_state, action_pred, true_action, c_t, print_loss=False):
    #loss-function weights
    Lambda_1 = 1  
    Lambda_2 = 1
    
    beta = 0.1

    L_BC = Lambda_2 * loss_fn(action_pred, true_action)
    L_ODC = Lambda_1 * loss_fn(next_state_pred, true_next_state)

    qy = c_t
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    #if print_loss:
    #    print('L_ODC: {} L_BC: {} Reg: {}'.format(L_ODC, L_BC, beta*KLD))
    
    return L_BC, L_ODC, beta*KLD


def train(epoch):
    model.train()
    train_loss = 0

    global temp, epochs, traj_length
    global hard
    global temp_plot, next_state_pred_plot, action_pred_plot, c_t_plot

    # store current temperature value
    current_temp = temp

    c_t_initial = torch.Tensor([[1,0]])
    c_t_stored = c_t_initial.view(-1, 1, latent_dim*categorical_dim)
    c_t_stored = c_t_stored.repeat(traj_length+1, 1, 1)

    # initialize variables and create arrays to store plotting data
    i=0
    L_BC_epoch, L_ODC_epoch, Reg_epoch, L_TSR_epoch = 0, 0, 0, 0
    action_pred_plot = np.zeros((traj_length-1, action_dim))
    # the /2 terms accounts for only predicting the next state (not previous)
    next_state_pred_plot = np.zeros((traj_length-1, int(state_dim/2)))
    c_t_plot = np.zeros((traj_length-1, latent_dim*categorical_dim))

    # loops until traj_length-1 because we need to use the next state as true_next_state
    for k in range(traj_length-1):
        data = traj_data[k]
        state, action = data[:,:state_dim], data[:,state_dim:]
        i += 1
        # print(i)
        # print('state: ', state)
        # print(' action: ', action)

        optimizer.zero_grad()
        c_prev = c_t_stored[i-1]

        # # TEMP: fixing value of c_t to evaluate other aspects of the model
        # c_prev = torch.Tensor([[1,0]])

        # predict next actions, states, and options
        action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)

        # # TEMP: fixing value of c_t to evaluate other aspects of the model
        # c_t = torch.Tensor([[1,0]])

        # store predictions for plotting after training
        action_pred_plot[i-1] = action_pred.detach().numpy()
        next_state_pred_plot[i-1] = next_state_pred.detach().numpy()
        c_t_plot[i-1] = c_t.detach().numpy()

        c_t_stored[i] = c_t
        true_next_state = traj_data[k+1][:,:int(state_dim/2)]

        #loss = loss_function(next_state_pred, state, action_pred, action, c_t)
        L_BC, L_ODC, Reg = loss_function(next_state_pred,true_next_state,action_pred,action,c_t)
        L_TSR = 0
        #L_TSR = 1-torch.dot(c_t_stored[i].squeeze(),c_t_stored[i-1].squeeze())

        L_BC_epoch += L_BC.item()
        L_ODC_epoch += L_ODC.item()
        Reg_epoch += Reg.item()
        #L_TSR_epoch += L_TSR.item()

        loss = L_BC + L_ODC + Reg + L_TSR
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()

    # update temperature value
    temp = np.maximum(temp * np.exp(-ANNEAL_RATE*epoch), temp_min)

    # print info
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss/i))
    print('L_ODC: {} L_BC: {} Reg: {}'.format(L_ODC_epoch/i, L_BC_epoch/i, Reg_epoch/i))
    print('L_TSR: {}'.format(L_TSR_epoch/i))
    print('current_temp: {}'.format(current_temp))

    return train_loss/i, L_BC_epoch/i, L_ODC_epoch/i, Reg_epoch/i, current_temp

def run():
    # create array to store loss values for plotting
    # format: epoch | train_loss | L_BC_epoch | L_ODC_epoch | Reg_epoch | temp
    loss_plot = np.zeros((epochs,6))

    # training loop
    for epoch in range(1, epochs + 1):
        # train
        train_loss, L_BC_epoch, L_ODC_epoch, Reg_epoch, current_temp = train(epoch)

        # store loss values
        loss_plot[epoch-1] = epoch, train_loss, L_BC_epoch, L_ODC_epoch, Reg_epoch, current_temp

    # plot predicted values for the last epoch of training
    plt.figure()
    plt.title('Evaluate Option-conditioned Policy')
    plt.plot(traj_data.numpy()[:,:,4], 'b-', label='a0')
    plt.plot(traj_data.numpy()[:,:,5], 'r-',  label='a1')
    plt.plot(action_pred_plot[:,0], 'b--', label='a0_pred')
    plt.plot(action_pred_plot[:,1], 'r--', label='a1_pred')
    plt.legend()
    plt.savefig('eval_policy.png')

    plt.figure()
    plt.title('Evaluate Option-conditioned Dynamics')
    plt.plot(traj_data.numpy()[:,:,0], 'b-', label='s0')
    plt.plot(traj_data.numpy()[:,:,1], 'r-',  label='s1')
    plt.plot(next_state_pred_plot[:,0], 'b--', label='s0_pred')
    plt.plot(next_state_pred_plot[:,1], 'r--', label='s1_pred')
    plt.legend()
    plt.savefig('eval_dynamics.png')

    plt.figure()
    plt.title('Evaluate Option Inference')
    plt.plot(np.argmax(c_t_plot[:,:categorical_dim], axis=1), 'b-')
    plt.savefig('eval_options.png')

    # plot losses
    plt.figure()
    plt.title('Training Loss')
    plt.plot(loss_plot[:,0], loss_plot[:,1], label='train_loss')
    plt.plot(loss_plot[:,0], loss_plot[:,2], label='L_BC_epoch')
    plt.plot(loss_plot[:,0], loss_plot[:,3], label='L_ODC_epoch')
    plt.plot(loss_plot[:,0], loss_plot[:,4], label='Reg_epoch')
    plt.legend()
    plt.savefig('training_loss.png')

    # plot temp
    plt.figure()
    plt.title('Temperature')
    plt.plot(loss_plot[:,0], loss_plot[:,5])
    plt.savefig('temp_loss.png')

    plt.show()

if __name__ == '__main__':
    run()
