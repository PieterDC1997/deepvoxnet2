#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:41:59 2023

@author: pieter
"""

import subprocess
import json
import os
import sys

target_venv_name = 'fip'

# get the names of the environments, assert that fip environment exists
result = subprocess.run(['conda', 'env', 'list', '--json'], capture_output=True, text=True)
env_list = json.loads(result.stdout)
env_names = [os.path.basename(os.path.normpath(env)) for env in env_list['envs']]
assert target_venv_name in env_names, "fip environment does not exist, which means the installation was not done correctly. Run the init.sh script of the fip pipeline in your terminal!"


# Check if the currently active environment matches the target environment
current_env = os.path.basename(os.path.normpath(sys.prefix))

assert target_venv_name == current_env, "You are currently not working in the fip virtual environment. Close your script/python/terminal, open a terminal and type  'conda activate fip', and start from scratch. Alternatively, when you are not familiar with coding and python, just re-run the run.sh script in the fip pipeline (see documentation). "     

check_environment = current_env

