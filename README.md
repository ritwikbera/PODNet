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
