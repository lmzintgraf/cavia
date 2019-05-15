import os
import random
import hashlib

import numpy as np
import torch


def seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I dont know where I am; please specify a path for saving results.')


def get_path_from_config(config):
    """ Returns a unique path from a config dict. """
    config_str = str(config)
    path = hashlib.md5(config_str.encode()).hexdigest()
    return path
