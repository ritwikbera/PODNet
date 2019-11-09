import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from circleworld import normalize, denormalize, gen_circle_traj
from gumbel import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from utils import to_categorical

import pickle

# -----------------------------------------------
# Random seeds
seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

# -----------------------------------------------
# Experiment hyperparameters
PLOT_RESULTS = True
epochs = 20
hard = False 

# from Directed-InfoGAIL
temp = 5.0
temp_min = 0.1
ANNEAL_RATE = 0.003
learning_rate = 1e-4
mlp_hidden = 32

# -----------------------------------------------
# Environment
# env_name = 'CircleWorld'
env_name = 'PerimeterDef'

if env_name == 'CircleWorld':
    state_dim = 4 # (x_t, y_t, x_prev, y_prev) of circle
    action_dim = 2
    latent_dim = 1 # has a similar effect to multi-head attention
    categorical_dim = 2 # number of options to be discovered

    # generate normalized trajectory data
    traj_data, true_segments_int = gen_circle_traj(
        r_init=1, n_segments=2, plot_traj=False, save_csv=False)
    traj_length = len(traj_data)

    # convert trajectory segments to pytorch format after one-hot encoding
    true_segments = torch.from_numpy(to_categorical(true_segments_int))
    c_t_stored = true_segments.view(traj_length,1,categorical_dim)

elif env_name == 'PerimeterDef':
    state_dim = 32 # includes previous states
    action_dim = 16
    latent_dim = 1 # has a similar effect to multi-head attention
    categorical_dim = 4 # number of options to be discovered

    # load dataset
    dataset = np.genfromtxt('data/sample_robots.csv', delimiter=',')
    traj_data, true_segments_int = dataset[:,:state_dim+action_dim], dataset[:,-1]
    traj_length = traj_data.shape[0]

    # normalize states and actions and convert to pytorch format
    traj_data, traj_data_mean, traj_data_std = normalize(traj_data)
    traj_data = torch.Tensor(np.expand_dims(traj_data, axis=1))

    # convert trajectory segments to pytorch format after one-hot encoding
    true_segments = torch.from_numpy(to_categorical(true_segments_int))
    c_t_stored = true_segments.view(traj_length,1,categorical_dim)


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
        self.dropout = nn.Dropout(p=0.3)
        self.use_dropout_encode = False
        self.use_dropout_dynamics = False
        self.use_dropout_policy = False

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        if self.use_dropout_encode:
            h1 = self.dropout(h1)
        h2 = self.relu(self.fc2(h1))
        if self.use_dropout_encode:
            h2 = self.dropout(h2)
        return self.fc3(h2)
        
    def decode_next_state(self, s_t, c_t, c_prev):
        z = torch.cat((s_t, c_t, c_prev), -1)
        h1 = self.relu(self.fc4(z))
        if self.use_dropout_dynamics:
            h1 = self.dropout(h1)
        h2 = self.relu(self.fc5(h1))
        if self.use_dropout_dynamics:
            h2 = self.dropout(h2)
        return self.fc6(h2)

    def decode_action(self, s_t, c_t, c_prev):
        z = torch.cat((s_t, c_t, c_prev), -1)
        h1 = self.relu(self.fc7(z))
        if self.use_dropout_policy:
            h1 = self.dropout(h1)
        h2 = self.relu(self.fc8(h1))
        if self.use_dropout_policy:
            h2 = self.dropout(h2)
        return self.fc9(h2)

    def forward(self, s_t, c_prev, temp, hard):
        x = torch.cat((s_t, c_prev), -1)
        q = self.encode(x.view(-1, state_dim + latent_dim*categorical_dim))
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        c_t = gumbel_softmax(q_y, temp, latent_dim, categorical_dim, hard)

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
    
    beta = 0.001

    L_BC = Lambda_2 * loss_fn(action_pred, true_action)
    L_ODC = Lambda_1 * loss_fn(next_state_pred, true_next_state)

    qy = c_t
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()
    
    return L_BC, L_ODC, beta*KLD


def train(epoch):
    model.train()
    train_loss = 0

    global temp, epochs, traj_length
    global hard

    # store current temperature value
    current_temp = temp

    # initialize variables and create arrays to store plotting data
    i = 0
    L_BC_epoch, L_ODC_epoch, Reg_epoch, L_TSR_epoch = 0, 0, 0, 0

    # loops until traj_length-1 because we need to use the next state as true_next_state
    for k in range(traj_length-1):
        data = traj_data[k]
        state, action = data[:,:state_dim], data[:,state_dim:]
        i += 1

        optimizer.zero_grad()
        c_prev = c_t_stored[i-1]

        # predict next actions, states, and options
        action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)
        c_t_stored[i] = c_t
        true_next_state = traj_data[k+1][:,:int(state_dim/2)]

        #loss = loss_function(next_state_pred, state, action_pred, action, c_t)
        L_BC, L_ODC, Reg = loss_function(next_state_pred,true_next_state,action_pred,action,c_t)
        L_TSR = 0
        # L_TSR = 1-torch.dot(c_t_stored[i].squeeze(),c_t_stored[i-1].squeeze())

        L_BC_epoch += L_BC.item()
        L_ODC_epoch += L_ODC.item()
        Reg_epoch += Reg.item()
        L_TSR_epoch += 0#L_TSR.item()

        loss = L_BC + L_ODC + Reg + L_TSR
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()

    # update temperature value
    temp = np.maximum(temp * np.exp(-ANNEAL_RATE*epoch), temp_min)

    # print info
    print('====> Epoch: {}/{} Average loss: {:.4f}'.format(epoch, epochs, train_loss/i))
    print('L_ODC: {:.4f} L_BC: {:.4f} Reg: {:.5f} L_TSR: {:.5f} temp: {:.2f}'.format(
        L_ODC_epoch/i, L_BC_epoch/i, Reg_epoch/i, L_TSR_epoch/i, current_temp))

    return train_loss/i, L_BC_epoch/i, L_ODC_epoch/i, Reg_epoch/i, current_temp, L_TSR_epoch/i

def eval():
    model.eval()

    global temp_min, hard
    global temp_plot, next_state_pred_plot, action_pred_plot, c_t_plot
    current_temp = temp_min

    # initialize variables and create arrays to store plotting data
    i=0
    action_pred_plot = np.zeros((traj_length-1, action_dim))
    # the /2 terms accounts for only predicting the next state (not previous)
    next_state_pred_plot = np.zeros((traj_length-1, int(state_dim/2)))
    c_t_plot = np.zeros((traj_length-1, latent_dim*categorical_dim))

    # loops until traj_length-1 because we need to use the next state as true_next_state
    for k in range(traj_length-1):
        data = traj_data[k]
        state, action = data[:,:state_dim], data[:,state_dim:]

        i += 1
        c_prev = c_t_stored[i-1]

        # predict next actions, states, and options
        action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)

        # store predictions for plotting after training
        action_pred_plot[i-1] = action_pred.detach().numpy()
        next_state_pred_plot[i-1] = next_state_pred.detach().numpy()
        c_t_plot[i-1] = c_t.detach().numpy()
        c_t_stored[i] = c_t

    return traj_data, true_segments

def run():
    # create array to store loss values for plotting
    # format: epoch | train_loss | L_BC_epoch | L_ODC_epoch | Reg_epoch | temp | L_TSR_epoch
    loss_plot = np.zeros((epochs,7))

    # training loop
    for epoch in range(1, epochs + 1):
        # train
        train_loss, L_BC_epoch, L_ODC_epoch, Reg_epoch, current_temp, L_TSR_epoch = train(epoch)

        # store loss values
        loss_plot[epoch-1] = epoch, train_loss, L_BC_epoch, L_ODC_epoch, Reg_epoch, current_temp, L_TSR_epoch

    # evaluate model
    traj_data, true_segments = eval()

    # plot predicted values for the last epoch of training
    # policy
    if PLOT_RESULTS:
        plt.figure()
        plt.title('Evaluate Option-conditioned Policy')
        for i in range(action_dim):
            p = plt.plot(traj_data.numpy()[:,:,state_dim+i], '-', label='a{}'.format(i))
            plt.plot(action_pred_plot[:,i], '--', color=p[0].get_color(), label='a{}_pred'.format(i))
        plt.legend()
        plt.savefig('eval_policy.png')

        # dynamics
        plt.figure()
        plt.title('Evaluate Option-conditioned Dynamics')
        for i in range(int(state_dim/2)):
            p = plt.plot(traj_data.numpy()[:,:,i], '-', label='s{}'.format(i))
            plt.plot(next_state_pred_plot[:,i], '--', color=p[0].get_color(), label='s{}_pred'.format(i))
        plt.legend()
        plt.savefig('eval_dynamics.png')

        # inference
        # postprocess true labels
        plt.figure()
        plt.title('Evaluate Option Inference')
        plt.plot(true_segments_int,'D',alpha=0.5,label='truth')
        plt.plot(np.argmax(c_t_plot[:,:categorical_dim], axis=1), 'k.', label='pred')
        plt.legend()
        plt.savefig('eval_options.png')

        # plot losses
        plt.figure(figsize=[12,8])
        plt.suptitle('Training Loss')
        plt.subplot(321)
        plt.plot(loss_plot[:,0], loss_plot[:,1], label='train_loss')
        plt.legend()
        plt.subplot(322)
        plt.plot(loss_plot[:,0], loss_plot[:,2], label='L_BC_epoch')
        plt.legend()
        plt.subplot(323)
        plt.plot(loss_plot[:,0], loss_plot[:,3], label='L_ODC_epoch')
        plt.legend()
        plt.subplot(324)
        plt.plot(loss_plot[:,0], loss_plot[:,4], label='Reg_epoch')
        plt.legend()
        plt.subplot(325)
        plt.plot(loss_plot[:,0], loss_plot[:,6], label='L_TSR_epoch')
        plt.legend()
        plt.subplot(326)
        plt.plot(loss_plot[:,0], loss_plot[:,5], label='temp')
        plt.legend()
        plt.savefig('training_loss.png')

        plt.show()

if __name__ == '__main__':
    run()
