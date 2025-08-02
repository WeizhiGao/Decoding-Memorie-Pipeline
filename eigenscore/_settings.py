import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = '/proj/csc266/scratch/wgao23/code/dmp/eigenscore'
MODEL_PATH = '/proj/csc266/scratch/wgao23/.cache/huggingface/offline/models'
DATA_FOLDER = '/proj/csc266/scratch/wgao23/.cache/huggingface/datasets'
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'log')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
