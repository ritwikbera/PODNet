import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import csv
import seaborn as sns
sns.set(style='whitegrid')

from utils import normalize, denormalize

def gen_circle_traj(r_init, n_segments, plot_traj=False, save_csv=False, save_csv_addr=None):
    state = []
    true_segments = []
    r = r_init
    last_fin = 0
    next_fin = 0
    direction = 1

    for j in range(n_segments):
        # generate size of segment
        fin = np.random.randint(low=30,high=150)
        print('Segment #{}: {} degrees'.format(j, fin))

        # randomize direction
        direction *= -1
        next_fin += fin*direction

        # prepare indexes to generate segments
        start_idx = last_fin
        end_idx = next_fin

        print(start_idx)
        print(end_idx)

        # generate states
        for theta in np.arange(start_idx, end_idx, direction):
            x = r*np.cos(theta*np.pi/180)
            y = r*np.sin(theta*np.pi/180)
            state.append([x, y])
            true_segments.append(direction)
            r *= 0.99

        last_fin = end_idx

    state = np.array(state)
    true_segments = np.clip(np.array(true_segments),0,1)

    # concatenates next states to current ones so the array of states becomes
    # (x_t, y_t, x_prev, y_prev) of circle
    true_segments = true_segments[1:]
    next_states = state[1:,:]
    prev_states = state[:-1,:]
    state_ = state
    state = np.hstack((next_states, prev_states))

    action = []
    # action.append([0,0])
    for i in range(1, len(state_)):
        action.append(state_[i,:2]-state_[i-1,:2])
    action = np.array(action)*10
    # smooth out abrupt changes in states
    # action = np.clip(action,-.0175,.0175)

    # check if have the same number of samples for states and actions
    print('Generated trajectory with {} samples.'.format(state.shape[0]))
    assert state.shape[0] == action.shape[0]

    # plot generated trajectories
    if plot_traj:
        plt.figure()
        plt.title('Generated States')
        plt.plot(state[:,0], label='s0')
        plt.plot(state[:,1], label='s1')
        plt.plot(state[:,2], label='s2')
        plt.plot(state[:,3], label='s3')
        plt.legend()

        plt.figure()
        plt.title('Generated Actions')
        plt.plot(action[:,0], 'o', label='a0')
        plt.plot(action[:,1], 'o', label='a1')
        plt.legend()

        # plt.show()
    
    # normalize generated data
    traj_data, traj_data_mean, traj_data_std = normalize(np.hstack((state, action)))

    # save generated trajectory in a csv file
    if save_csv:
        if save_csv_addr == None:
            save_csv_addr = 'circle_traj.csv'

        with open(save_csv_addr, mode='w') as csv_file:
            fieldnames = ['x_t','y_t','x_t-1','y_t-1','a_x','a_y','direction']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
    
        with open(save_csv_addr, mode='ab') as csv_file:
            np.savetxt( \
                csv_file, np.hstack((traj_data, true_segments.reshape(-1,1))), delimiter=',')

    return torch.Tensor(np.expand_dims(traj_data, axis=1)), traj_data_mean, traj_data_std, true_segments

if __name__ == "__main__":
    # traj_data,_,_,_ = gen_circle_traj( \
    #     r_init=100, n_segments=2, plot_traj=True, save_csv=True)
    # traj_length = len(traj_data)

    for i in range(20):
        n_segments = np.random.randint(low=2,high=4)
        traj_data,_,_,_ = gen_circle_traj( \
         r_init=10, n_segments=n_segments, plot_traj=True, save_csv=True, save_csv_addr='circle_{}.csv'.format(i))
