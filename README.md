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

- Run a single experiment with ```python train.py --epochs=30 --launch_tb=True --log_dir=<your_log_directory_name>```. Set ```launch_tb=True``` to launch tensorboard once training is completed

- Note that train.py will delete any pre-existing folder with the same name as your provided log_dir. Rename your log_dir after each run, accordingly.

- For hyperparameter search, setup the experiment wise configurations in the **experiments.json** file. Once all experiments have been run visualise results by running ```tensorboard --logdir=experiments\experiment<index>\tensorboard ```

- Make sure that the root PODNet folder does not have any pre-existing **experiments** folder before running ```search_hparams.py```. Rename you **experiments** folder after each  run, accordingly.

- In the **v2** folder, run ```python train.py --epochs 300``` to train and ```python eval.py``` to evaluate. Right now hardcode to work only with CircleWorld data for debugging purposes. Please check **v1** for previous, more general and not parallelized, model. 

# Citation

If you use this repository, please cite our work:  
```
@misc{bera2019podnet,
    title={PODNet: A Neural Network for Discovery of Plannable Options},
    author={Ritwik Bera and Vinicius G. Goecks and John Valasek and Nicholas R. Waytowich},
    year={2019},
    eprint={1911.00171},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
