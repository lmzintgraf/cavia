from collections import OrderedDict


def get_default_configs_cavia():
    return OrderedDict({
        'method': 'cavia',
        #
        'task': 'miniimagenet',
        'imsize': 84,
        #
        'n_iter': 60000,
        'seed': 123,
        #
        'tasks_per_metaupdate': 16,  # number of tasks per meta-update
        #
        'n_way': 5,
        'k_shot': 1,  # number of datapoints to learn from
        'k_query': 15,
        #
        'lr_inner': 1.0,
        'lr_meta': 0.001,
        #
        'num_grad_steps_inner': 2,
        'num_grad_steps_eval': 2,
        #
        # --- network ---
        #
        'model': 'cnn',
        'film': True,
        'max_pool': True,
        'num_filters': 32,
        'initialisation': 'kaiming',  # standard (uses pytorch default), xavier, kaiming
        #
        # context parameters
        'num_context_params': [0, 0, 100, 0, 0],
        'num_film_hidden_layers': 0,
        #
        'first_order': False,
        #
        'batchnorm_at_films': True,
    })
