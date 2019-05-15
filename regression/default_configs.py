from collections import OrderedDict


def get_default_config_cavia(task):
    """
    :param task:    (str) sine or celeba
    :return:
    """

    configs = OrderedDict({
        'method': 'cavia',
        'task': task,
        #
        'n_iter': 50000,
        'seed': 42,
        #
        'tasks_per_metaupdate': 25,  # number of tasks per meta-update
        #
        'k_meta_train': 10,  # data points in task training set (during meta training, inner loop)
        'k_meta_test': 10,  # data points in task test set (during meta training, outer loop)
        'k_shot_eval': 10,  # data points in task training set (during evaluation)
        #
        'lr_inner': 1.0,
        'lr_meta': 0.001,
        #
        'num_inner_updates': 1,
        #
        'num_context_params': 5,
        'n_hidden': [40, 40],
        #
        'first_order': False,
        #
    })

    if task == 'celeba':
        configs['order_pixels'] = True
    else:
        configs['order_pixels'] = None

    return configs


def get_default_config_maml(task):
    configs = OrderedDict({
        'method': 'maml',
        'task': task,  # sine, celeba
        #
        'n_iter': 50000,
        'seed': 42,
        #
        'tasks_per_metaupdate': 25,
        #
        'k_meta_train': 10,  # data points in task training set (during meta training, inner loop)
        'k_meta_test': 10,  # data points in task test set (during meta training, outer loop)
        'k_shot_eval': 10,  # data points in task training set (during evaluation)
        #
        'lr_inner': 0.01,
        'lr_meta': 0.001,
        #
        'num_inner_updates': 1,
        #
        'num_context_params': 0,  # additional biases
        'n_hidden': [40, 40],
        #
        'first_order': False,
    })

    if task == 'celeba':
        configs['order_pixels'] = True
    else:
        configs['order_pixels'] = None

    return configs
