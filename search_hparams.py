import argparse
import os
from subprocess import check_call
import sys
import multiprocessing as mp 
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/')

def launch_training_job(parent_dir, data_dir, job_name, params):

    model_dir = os.path.join(parent_dir, job_name)
    params['log_dir'] = model_dir
    print_config(params)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    json_path = os.path.join(model_dir, 'params.json')
    
    with open(json_path, 'w') as outfile:
        json.dump(params, outfile)
    
    cmd = f"python3 train.py --use_json=True --json_addr={model_dir}/params.json"
    
    print(cmd)
    check_call(cmd, shell=True)

def print_config(params):
    for key, value in params.items():
        print('{} : {}'.format(key, value))
    print('\n ---------- \n')

if __name__ == "__main__":
    args = parser.parse_args()
    experiments = json.load(open('experiments.json'))
    parent_dir = 'experiments'
    os.makedirs(parent_dir, exist_ok=True)

    for experiment in experiments:
        params = experiments[experiment]
        job_name = experiment
        launch_training_job(parent_dir, args.data_dir, job_name, params)
