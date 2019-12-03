import numpy as np 
import os
import pandas as pd
import math

input_file = 'minigrid_keydoor18.csv'
output_dir = 'minigrid'

def normalize(df):
    features = ['x_t', 'y_t', 'heading_t', 'key_x', 'key_y', 'goal_x', 'goal_y', 'door_x', 'door_y']
    features = ['x_t', 'y_t', 'heading_t']
    result = df.copy()
    # df['key_x'] = pd.to_numeric(df['key_x'])
    print(df.dtypes)
    for feature_name in features:
        mean = np.ceil(np.mean(df[feature_name]))
        print(mean)
        std = np.std(df[feature_name])
        std = 1
        result[feature_name] = (df[feature_name]-mean)
    return result

file = pd.read_csv(input_file)

cols = file.columns.tolist()
print(cols)
cols = cols[:2]+ cols[3:] + cols[2:3]
file = file[cols]

old_header = ['episode','time_step','x','y','heading', 'action']
new_header = ['episode', 't', 'x_t', 'y_t', 'heading_t', 'a_1']

changes = dict(zip(old_header, new_header))

file = file.rename(changes, axis=1)

file = normalize(file)

files = file.groupby('episode')
print('Number of episodes {}'.format(files.ngroups))

print(file)

os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
files.apply(lambda x: x.to_csv('minigrid{}.csv'.format(x.name), index=False))

traj = pd.read_csv('minigrid0.csv').set_index('t')
print('First Episode \n {}'.format(traj))
print('First Episode states only\n {}'.format(traj.loc[:,'x_t':'a_1']))

print(np.array(traj.loc[:,'x_t':'a_1'])[:,:-1])