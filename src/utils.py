import os 
import torch
import numpy as np
import random
from loguru import logger
        
def print_args(args):
    for k,v in vars(args).items():
        logger.info(f'{k}: {v}')
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False