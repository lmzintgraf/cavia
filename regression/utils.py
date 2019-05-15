import hashlib
import os
import pickle
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_path_from_config(config):
    """ Returns a unique path from a config dict. """
    config_str = str(config)
    path = hashlib.md5(config_str.encode()).hexdigest()
    return path


# ------------------ experiment result management ------------------

def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I dont know where I am; please specify a path for saving results.')
