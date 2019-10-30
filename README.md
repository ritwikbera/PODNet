# PODNet
PyTorch code for PODNet

# Installation and dependencies

## Docker

To be written.

## Virtual environments

Install _pip_ and _virtualenv_ to handle all Python3 dependencies:  
```sudo apt-get install python3-pip```  
```python3 -m pip install --user virtualenv```  

Create a new virtual environment:  
```python3 -m venv ~/venvs/PODNet```

Activate the new environment and install dependencies:  
```source ~/venvs/PODNet/bin/activate```  
```pip install wheel```  
```pip install -r requirements.txt```

# Instructions
- Run CatVAE.py
- Once dataset is downloaded, set download = False in the DataLoaders.

## Collect human trajectories

Run ```python collect_trajs.py``` and close environment window when finished collecting data. 
Default actions are:
- Left arrow: turns left  
- Right arrow: turns right  
- Up arrow: moves forward  

Data as saved as csv file in the data folder. Each row corresponds to a data sample and columns represent episode number, 
time step number of the current episode, action taken, and the rest are the observations stored as a flat array.
