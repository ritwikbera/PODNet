
import os
import pandas as pd

input_file = 'minigrid20.csv'
output_dir = 'minigrid'
header = ['episode', 't', 'a_t', 'x_t', 'y_t', 'heading_t']
file = pd.read_csv(input_file)
files = file.groupby('episode')
print('Number of episodes {}'.format(files.ngroups))
os.makedirs(output_dir, exist_ok=True)
os.chdir(output_dir)
files.apply(lambda x: x.to_csv('minigrid{}.csv'.format(x.name), index=False, header=False))

traj = pd.read_csv('minigrid0.csv', names=header).set_index('t')

print('First Episode \n {}'.format(traj))
