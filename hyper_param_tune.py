"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################


import os
import sys
import importlib
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from os.path import join, basename, exists

from dataloader.DataLoaders import *
from modules.losses import *
from datetime import datetime


# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

TRAIN_POOL_SIZE = 85898
LRS = [0.0005, 0.0001, 0.00005, 0.00001, 0.000005]
WDS = [0.02, 0.002, 0.0002, 0.00002, 0.000002]
#torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="json config file name", default="params_hyper_tune.json")
parser.add_argument('-mode', action='store', dest='mode', default='train', help='"eval" or "train" mode')
parser.add_argument('-exp', action='store', dest='exp', default='exp_msg_chn',
                    help='Experiment name as in workspace directory')
parser.add_argument('-chkpt', action='store', dest='chkpt', default=None,  nargs='?',   # None or number
#parser.add_argument('-chkpt', action='store', dest='chkpt', default="/PATH/TO/YOUR/CHECKPOINT_FILE.pth.tar",
                    help='Checkpoint number to load')

parser.add_argument('-set', action='store', dest='set', default='selval', type=str, nargs='?',
                    help='Which set to evaluate on "val", "selval" or "test"')
args = parser.parse_args()

# Path to the workspace directory
training_ws_path = 'workspace/'
exp = args.exp
exp_dir = join(training_ws_path, exp)

# Add the experiment's folder to python path
sys.path.append(exp_dir)

# Read parameters file
with open(join(exp_dir, args.config), 'r') as fp:
    params = json.load(fp)

#choose train samples 
train_idx = np.random.choice(TRAIN_POOL_SIZE, size=params['train_on'], replace=False)
device = torch.device("cuda:" + params['gpu_id'] if torch.cuda.is_available() else "cpu")
params['train_idx'] = train_idx
for lr in LRS:
    for wd in WDS:
        params['lr'] = lr
        params['weight_decay'] = wd
        exp_dir = join(training_ws_path, exp)
        experiment = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        experiment += "_LR_" + str(lr) + "_WD_" + str(wd)
        exp_dir = join(exp_dir, experiment)
        if not exists(exp_dir):
            os.makedirs(join(exp_dir, 'tensorboard'))

        logger = SummaryWriter(join(exp_dir, 'tensorboard'))

        # Dataloader
        data_loader = params['data_loader'] if 'data_loader' in params else 'KittiDataLoader'
        dataloaders, dataset_sizes = eval(data_loader)(params)

        # Import the network file
        f = importlib.import_module('network_' + exp)
        model = f.network().to(device)#pos_fn=params['enforce_pos_weights']
        model = nn.DataParallel(model)

        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        # Import the trainer
        t = importlib.import_module('trainers.' + params['trainer'])

        mode = 'train'  # train    eval
        sets = ['train', 'selval']  # train  selval

        # Objective function
        objective = locals()[params['loss']]()

        # Optimize only parameters that requires_grad
        parameters = filter(lambda p: p.requires_grad, model.parameters())

        # The optimizer
        optimizer = getattr(optim, params['optimizer'])(parameters, lr=params['lr'],
                                                            weight_decay=params['weight_decay'])

        lr_decay = lr_scheduler.StepLR(optimizer, step_size=params['lr_decay_step'], gamma=params['lr_decay'])

        mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, lr_decay, dataloaders, dataset_sizes,
                                            workspace_dir=exp_dir, sets=sets, use_load_checkpoint=args.chkpt, logger=logger)


        net = mytrainer.train(params['num_epochs'])  #






