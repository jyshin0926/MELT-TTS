import os
import subprocess

# Set environment variables
os.environ['MASTER_PORT'] = '6081'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPUS_PER_NODE'] = '1'

# Configuration directory and name
config_dir = './'
config_name = 'base'

# Build the command to run
command = [
    'torchrun',
    '--nproc_per_node={}'.format(os.environ['GPUS_PER_NODE']),
    '--master_port={}'.format(os.environ['MASTER_PORT']),
    '../../train.py',
    '--config-dir={}'.format(config_dir),
    '--config-name={}'.format(config_name)
]

# Execute the command
subprocess.run(command)
