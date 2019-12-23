import torch
from torch import Tensor, nn, optim
from torch.utils.data import Dataset
import pandas as pd
import glob
import numpy as np

def pad_trajectory(trajectory, PAD_TOKEN, MAX_LENGTH):
    padded_traj = Tensor([PAD_TOKEN]).repeat(MAX_LENGTH, trajectory.size(-1))
    padded_traj[:trajectory.size(0)] = trajectory
    return padded_traj

def segment_traj(trajectory, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE):
    trajectory = pad_trajectory(trajectory, PAD_TOKEN, MAX_LENGTH)
    traj_segments = torch.split(trajectory, SEGMENT_SIZE, dim=-2)
    traj_segment_batch = torch.stack(traj_segments).view(-1, SEGMENT_SIZE, trajectory.size(-1))
    return traj_segment_batch

def load_segment_stack(trajectory, PAD_TOKEN=0, MAX_LENGTH=2048, SEGMENT_SIZE=512):
    states = segment_traj(trajectory, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE)
    true_next_states = segment_traj(trajectory[1:], PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE)
    return states, true_next_states

class RoboDataset(Dataset):
    def __init__(self, PAD_TOKEN, MAX_LENGTH, SEGMENT_SIZE, root_dir='data/'):
        self.root_dir = root_dir
        self.PAD_TOKEN = PAD_TOKEN
        self.MAX_LENGTH = MAX_LENGTH
        self.SEGMENT_SIZE = SEGMENT_SIZE

    def __len__(self):
        return len(glob.glob(self.root_dir+'*'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file = glob.glob(self.root_dir+'*')[idx]
        traj_file = pd.read_csv(file)
        trajectory = Tensor(np.array([traj_file['x_t'], traj_file['y_t']]).T)
        return load_segment_stack(trajectory, self.PAD_TOKEN, self.MAX_LENGTH, self.SEGMENT_SIZE)

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