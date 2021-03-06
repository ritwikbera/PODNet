''' podnet.py
profile: python -m cProfile -o podnet.prof podnet.py
viz: snakeviz podnet.prof
'''
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

from circleworld import gen_circle_traj
from gumbel import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from utils import to_categorical, normalize, denormalize

import pickle

class OptionEncoder(nn.Module):
    def __init__(self, output_size, input_size, hidden_layer_size=32, num_layers=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size),
                            torch.zeros(num_layers,1,self.hidden_layer_size))

    def forward(self, state, c_prev):
        c_prev = c_prev.view(c_prev.size(0), -1)
        state = state.view(state.size(0), -1)
        lstm_out, self.hidden_cell = self.lstm(torch.cat((state,c_prev),-1).unsqueeze(0), 
            (self.hidden_cell[0].detach(), self.hidden_cell[1].detach()))
        predictions = self.linear(lstm_out.view(1,1,-1))
        return predictions.squeeze(0)  #single length input so squeeze out seq_len dimension

class PODNet(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, categorical_dim, mlp_hidden=32, use_recurrent=False):
        super(PODNet, self).__init__()

        # inputs parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.use_recurrent = use_recurrent    

        if use_recurrent:
            self.OptInference = OptionEncoder(output_size=latent_dim*categorical_dim, 
                input_size=state_dim+latent_dim*categorical_dim)
        else:
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

        self.loss_fn = torch.nn.MSELoss()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)
        self.use_dropout_encode = False
        self.use_dropout_dynamics = False
        self.use_dropout_policy = False

    def recurrent_encode(self, s_t, c_prev):
        return self.OptInference(s_t, c_prev)

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
        if self.use_recurrent:
            q = self.recurrent_encode(s_t, c_prev)
        else:
            x = torch.cat((s_t, c_prev), -1)
            q = self.encode(x.view(-1, self.state_dim + self.latent_dim*self.categorical_dim))
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        c_t = gumbel_softmax(q_y, temp, self.latent_dim, self.categorical_dim, hard, device=self.device)

        action_pred = self.decode_action(s_t, c_t, c_prev)
        next_state_pred = self.decode_next_state(s_t, c_t, c_prev)

        return action_pred, next_state_pred, c_t


    def loss_function(self, next_state_pred, true_next_state, action_pred, true_action, c_t, print_loss=False):
        #loss-function weights
        Lambda_1 = 1  
        Lambda_2 = 1
        
        beta = 1e-4

        L_BC = Lambda_2 * self.loss_fn(action_pred, true_action)
        L_ODC = Lambda_1 * self.loss_fn(next_state_pred, true_next_state)

        qy = c_t
        log_ratio = torch.log(qy * self.categorical_dim + 1e-20)
        KLD = torch.sum(qy * log_ratio, dim=-1).mean()
        
        return L_BC, L_ODC, beta*KLD


def run(EVAL_MODEL, epochs, hard, exp_name, env_name, use_recurrent, n_trajs):
    # if available, use single GPU to train PODNet
    if torch.cuda.is_available():  
        device = torch.device("cuda:0")
    else:  
        device = torch.device("cpu")

    # GPU not working for recurrent case
    if use_recurrent:
        device = torch.device("cpu")
    print(f'[*] INFO: Using {device} device.')

    # parse env parameters
    if env_name == 'CircleWorld':
        state_dim = 4 # (x_t, y_t, x_prev, y_prev) of circle
        action_dim = 2
        latent_dim = 1 # has a similar effect to multi-head attention
        categorical_dim = 2 # number of options to be discovered

        # from Directed-InfoGAIL
        temp = 5.0
        temp_min = 0.1
        ANNEAL_RATE = 0.003
        learning_rate = 1e-4
        mlp_hidden = 32

        # generate normalized trajectory data
        traj_data, traj_data_mean, traj_data_std, true_segments_int = gen_circle_traj(
            r_init=1, n_segments=2, plot_traj=False, save_csv=True)
        traj_length = len(traj_data)

        # convert trajectory segments to pytorch format after one-hot encoding
        true_segments = torch.from_numpy(to_categorical(true_segments_int))
        c_initial = torch.Tensor([[1,0]])

    elif env_name == 'PerimeterDef':
        # Option labels:
        #   1 - CYCLIC PURSUIT
        #   2 - LEADER-FOLLOWER
        #   3 - FORMATION - CIRCLE
        #   4 - FORMATION - CONE
        state_dim = 32 # includes previous states
        action_dim = 16
        latent_dim = 1 # has a similar effect to multi-head attention
        categorical_dim = 4 # number of options to be discovered

        # from Directed-InfoGAIL
        temp = 5.0
        temp_min = 0.1
        ANNEAL_RATE = 0.003
        learning_rate = 1e-4
        mlp_hidden = 32

    # create PODNet
    model = PODNet(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        categorical_dim=categorical_dim,
        mlp_hidden=mlp_hidden,
        use_recurrent=use_recurrent
    ).to(device)
    model.device = device
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
    # create array to store loss values for plotting
    # format: epoch | train_loss | L_BC_epoch | L_ODC_epoch | Reg_epoch | temp | L_TSR_epoch
    loss_plot = np.zeros((epochs,7))

    # training loop
    start_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        # --------------------------------------------------------------------
        # TRAIN
        print(f'*** EPOCH {epoch}/{epochs}. Using {n_trajs} trajectories ***')
        start_epoch = time.time()

        # store current temperature value
        current_temp = temp

        # initialize variables and create arrays to store plotting data
        i = 0
        train_loss, L_BC_epoch, L_ODC_epoch, Reg_epoch, L_TSR_epoch = 0, 0, 0, 0, 0

        # load it trajectory separetely. otherwise it won't fit on memory
        for j in range(n_trajs):
            if env_name == 'PerimeterDef':
                # load dataset
                file_name = f'robots/data/{j}_1dsf_log.csv'
                dataset = np.genfromtxt(file_name, delimiter=',')
                traj_data, true_segments_int = dataset[:,:state_dim+action_dim], dataset[:,-1]
                traj_length = traj_data.shape[0]
                print('Loading {} dataset. {} robot data points.'.format(file_name, traj_length))
                
                # normalize states and actions and convert to pytorch format
                traj_data, traj_data_mean, traj_data_std = normalize(traj_data)
                traj_data = torch.Tensor(np.expand_dims(traj_data, axis=1))

                # convert trajectory segments to pytorch format after one-hot encoding
                true_segments = torch.from_numpy(to_categorical(true_segments_int))
                c_initial = torch.Tensor([[1,0,0,0]])

            # send data to current device (CPU or GPU)
            traj_data = traj_data.to(device)
            true_segments = true_segments.to(device)
            c_prev = c_initial.to(device)

            # loops until traj_length-1 because we need to use the next state as true_next_state
            for k in range(traj_length-1):
                data = traj_data[k]
                state, action = data[:,:state_dim], data[:,state_dim:]
                i += 1

                optimizer.zero_grad()

                # predict next actions, states, and options
                action_pred, next_state_pred, c_t = model(state,c_prev,current_temp,hard)
                true_next_state = traj_data[k+1][:,:int(state_dim/2)]

                #loss = loss_function(next_state_pred, state, action_pred, action, c_t)
                L_BC, L_ODC, Reg = model.loss_function(next_state_pred,true_next_state,action_pred,action,c_t)
                L_TSR = 0
                # L_TSR = 1-torch.dot(c_t_stored[i].squeeze(),c_t_stored[i-1].squeeze())

                # save predicted c_t to use as previous c_t on next iteration
                #c_prev = c_t.detach()
                c_prev = Variable(c_t.data.clone(), requires_grad=False)

                # Propagates gradients after every data sample
                L_BC_epoch += L_BC.item()
                L_ODC_epoch += L_ODC.item()
                Reg_epoch += Reg.item()
                L_TSR_epoch += 0#L_TSR.item()

                loss = L_BC + L_ODC + Reg + L_TSR
                loss.backward(retain_graph=False)
                train_loss += loss.item()
                optimizer.step()

        # update temperature value
        temp = np.maximum(temp * np.exp(-ANNEAL_RATE*epoch), temp_min)

        # print info
        print('====> Epoch: {}/{} Average loss: {:.4f}'.format(epoch, epochs, train_loss/i))
        print('L_ODC: {:.4f} L_BC: {:.4f} Reg: {:.5f} L_TSR: {:.5f} temp: {:.2f}'.format(
            L_ODC_epoch/i, L_BC_epoch/i, Reg_epoch/i, L_TSR_epoch/i, current_temp))

        epoch_time = time.time()-start_epoch
        print('Epoch time: {:.2f} seconds. Estimated {:.2f} minutes or {:.2f} hours left to complete training.'.format(
            epoch_time, (epochs-epoch)*epoch_time/60, (epochs-epoch)*epoch_time/3600))

        # store loss values
        loss_plot[epoch-1] = epoch, train_loss/i, L_BC_epoch/i, L_ODC_epoch/i, Reg_epoch/i, current_temp, L_TSR_epoch/i

        # save last trained model
        torch.save({
            'env_name': env_name,
            'exp_name': exp_name,
            'temp_min': temp_min,
            'hard': hard,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'categorical_dim': categorical_dim,
            'latent_dim': latent_dim,
            'c_initial': c_initial,
            'loss_plot': loss_plot,
            'use_recurrent': use_recurrent,
            'model_state_dict': model.state_dict(),
        }, f'results/{exp_name}/{env_name}_ep{epoch}_trained.pt')

    print('[*] Total training time: {:.2f} minutes'.format((time.time()-start_train)/60))

    # evaluate model
    if EVAL_MODEL:
        print('[*] Running trained model on evaluation data.')
        os.system(f"python eval_podnet.py results/{exp_name}/{env_name}_trained.pt robots/data/198_1dsf_log.csv")


if __name__ == '__main__':
    # -----------------------------------------------
    # Random seeds
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------------------------
    # Experiment hyperparameters
    EVAL_MODEL = True
    epochs = 50
    hard = False 
    use_recurrent = False

    # -----------------------------------------------
    # Environment
    # exp_name = 'circle'
    # env_name = 'CircleWorld'
    # n_trajs = 1

    exp_name = '180_trajs_1dsf'
    env_name = 'PerimeterDef'
    n_trajs = 180

    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{exp_name}", exist_ok=True)

    # train PODNet
    run(EVAL_MODEL, epochs, hard, exp_name, env_name, use_recurrent, n_trajs)
