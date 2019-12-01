# Data

Stores important datasets and temp files after each experiment.

# Data Collection

## Minimalistic Gridworld Environment (MiniGrid)

This section describes how to collect human generated trajectories using the [Gym MiniGrid](https://github.com/maximecb/gym-minigrid]) environment. In the **v1** folder, run ```python collect_trajs.py``` and close environment window when finished collecting data. 
Default actions are:
- Left arrow: turns left  
- Right arrow: turns right  
- Up arrow: moves forward  

Data as saved as csv file in the data folder. Each row corresponds to a data sample and columns represent episode number, 
time step number of the current episode, action taken, and the rest are the observations stored as a flat array.

# Sample Data

- **circle_traj_header.csv**: one trajectory collected on the CircleWorld environment.

- **minigrid20.csv**: twenty human generated trajectories collected on the MiniGrid-Empty-Random-6x6-v0 environment.

- **key_door_21epis.csv**: twenty-one human generated trajectories collected on the MiniGrid-DoorKey-16x16-v0 environment.

- **perimeter_intruder.csv**: one trajectory of the intruder robot on the PerimeterDef environment (episode 2).

- **perimeter_defender.csv**: one trajectory of the defender robot on the PerimeterDef environment (episode 2).