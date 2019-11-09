""" lfo.py:
Machine learning code for classifying coordinated behaviors in a perrimiter defense task
for the learning from observations for multiagent systems project. 

__author__ = "Nicholas Waytowich, Vinicius G. Goecks"
__version__ = "1.0.0"
__date__ = "November 09, 2019"

"""
import csv
import re
import numpy as np
import scipy.io as sio
import scipy as sp
from scipy import signal
import random

from os import listdir
from os.path import isfile, join
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# copied from Keras to avoid tensorflor installation
# source: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


#############################################
# class for loading and pre-processing data
#############################################
class DataProcessor(object):
    def __init__(self, file_dir = '', files = [], randomize = 'False'):
        self.file_dir = file_dir
        self.files = files        
        
        # load data
        self.load_data()

        # process data for training
        self.process_data(randomize = randomize)
        
        
    def load_data(self):
        data = []
        self.behaviors = []
        
        for file in self.files:
            f= open(self.file_dir+file, 'rt')
            print('Loading ', self.file_dir+file)
            reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    
        
            lines=[]
            for line in reader:
                lines.append(line)
                 
            # decode data
            for i in range(0,len(lines),11):
                #print('processing sample {}'.format(int(i/10)))
                
                # new sample
                sample = dict()
                
                sample['label_gt'] = lines[i][0]
                self.behaviors.append(lines[i][0])
                
                sample['label'] = lines[i+1][0]
                #self.behaviors.append(lines[i][0])
                
                sample['real-state'] = np.concatenate((np.array(re.split(r'\t+', (lines[i+3][0]).rstrip('\t'))),
                np.array(re.split(r'\t+', (lines[i+4][0]).rstrip('\t')))))
                
                sample['imm-state'] = np.concatenate((np.array(re.split(r'\t+', (lines[i+6][0]).rstrip('\t'))),
                np.array(re.split(r'\t+', (lines[i+7][0]).rstrip('\t')))))
                
                sample['env-state'] = np.concatenate((np.array(re.split(r'\t+', (lines[i+9][0]).rstrip('\t'))),
                np.array(re.split(r'\t+', (lines[i+10][0]).rstrip('\t')))))
                
                data.append(sample)
            
        
        # construct class dictionary
        self.behaviors = list(set(self.behaviors))
        self.classes = dict()
        cidx = 0
        for b in self.behaviors:
            self.classes[b]=cidx
            cidx = cidx + 1
            
        # organize data
        assert len(self.classes) >= len(self.behaviors), 'Mismatch in number of behaviors'
        self.labels = np.zeros((len(data),1))
        self.labels_gt = np.zeros((len(data),1))
        self.x_real = np.zeros((len(data),16))
        self.x_imm = np.zeros((len(data),16))
        
        self.bad_idx = []
        for i in range(len(data)):
            try:
                self.labels[i] = self.classes[data[i]['label']]
                self.labels_gt[i] = self.classes[data[i]['label_gt']]
                self.x_real[i] = np.concatenate((data[i]['real-state'],data[i]['env-state']))
                self.x_imm[i] = np.concatenate((data[i]['imm-state'],data[i]['env-state']))
            except:
                #print('bad data')
                self.bad_idx.append(i)
        
        # remove bad data
        mask = np.ones(len(data), np.bool)
        mask[self.bad_idx] = 0
        self.labels = self.labels[mask]
        self.labels_gt = self.labels_gt[mask]
        self.x_real = self.x_real[mask,:]
        self.x_imm = self.x_imm[mask,:]
        
        self.y = to_categorical(self.labels)
        self.y_gt = to_categorical(self.labels_gt)
        
    def process_data(self, randomize = False):
        if randomize:
            rand_idx = np.random.permutation(len(self.y))
            self.y = self.y[rand_idx]
            self.y_gt = self.y_gt[rand_idx]
            self.labels = self.labels[rand_idx]
            self.labels_gt = self.labels_gt[rand_idx]
            self.x_real = self.x_real[rand_idx]
            self.x_imm = self.x_imm[rand_idx]
            
# Main starting point
if __name__ =='__main__':

    data_dir = 'data/'
    
    # data files
    datafiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    
    # load data
    train_data = DataProcessor(file_dir = data_dir, files = datafiles, randomize = False)

    # visualize data
    t_initial = 775
    t_limit = 1530
    plt.figure()
    plt.title('Sample Options')
    plt.plot(train_data.labels_gt[t_initial:t_limit])
    plt.ylabel('Option')
    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig('robot_options.png')

    plt.figure()
    plt.title('Sample States')
    plt.ylabel('Position Y')
    plt.xlabel('Position X')
    plt.plot(train_data.x_real[t_initial:t_limit,0], train_data.x_real[t_initial:t_limit,1], 'o', alpha=0.1, label='r1')
    plt.plot(train_data.x_real[t_initial:t_limit,2], train_data.x_real[t_initial:t_limit,3], 'o', alpha=0.1, label='r2')
    plt.plot(train_data.x_real[t_initial:t_limit,4], train_data.x_real[t_initial:t_limit,5], 'o', alpha=0.1, label='r3')
    plt.plot(train_data.x_real[t_initial:t_limit,6], train_data.x_real[t_initial:t_limit,7], 'o', alpha=0.1, label='r4')
    plt.plot(train_data.x_real[t_initial:t_limit,8], train_data.x_real[t_initial:t_limit,9], 'o', alpha=0.1, label='r5')
    plt.plot(train_data.x_real[t_initial:t_limit,10], train_data.x_real[t_initial:t_limit,11], 'o', alpha=0.1, label='r6')
    plt.plot(train_data.x_real[t_initial:t_limit,12], train_data.x_real[t_initial:t_limit,13], 'o', alpha=0.1, label='i1')
    plt.plot(train_data.x_real[t_initial:t_limit,14], train_data.x_real[t_initial:t_limit,15], 'o', alpha=0.1, label='i2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('robot_states.png')

    # save data sample
    # concatenate current and previous states
    next_states = train_data.x_real[1+t_initial:t_limit,:]
    prev_states = train_data.x_real[t_initial:t_limit-1,:]
    state = np.hstack((next_states, prev_states))
    # create action vectors (difference between states)
    action = state[:,0:16]-state[:,16:]

    # parse options
    option = train_data.labels_gt[1+t_initial:t_limit]

    # filter actions
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
    b, a = signal.ellip(4, 0.01, 120, 0.125)  # Filter to be applied
    f_action = np.copy(action)
    for i in range(f_action.shape[1]):
        f_action[:,i] = signal.filtfilt(b, a, f_action[:,i], padlen=50)

    plt.figure(figsize=[12,6])
    plt.suptitle('Example of Filtered Actions')
    for i in range(9):
        plt.subplot(int(331+i))
        plt.plot(action[:,i], '-', alpha=0.5)
        plt.plot(f_action[:,i], '--')
        plt.grid()
    plt.savefig('robot_actions.png', dpi=300)

    plt.show()

    # save data sample
    np.savetxt('sample_robots.csv', np.hstack((state,f_action,option)), delimiter=',')