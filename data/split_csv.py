import numpy as np 
import os
import pandas as pd

input_file = 'minigrid20.csv'
output_dir = 'minigrid'


file = pd.read_csv(input_file)

cols = file.columns.tolist()
print(cols)
cols = cols[:2]+ cols[3:] + cols[2:3]
file = file[cols]

old_header = ['episode','time_step','x','y','heading', 'action']
new_header = ['episode', 't', 'x_t', 'y_t', 'heading_t', 'a_1']

changes = dict(zip(old_header, new_header))

file = file.rename(changes, axis=1)

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