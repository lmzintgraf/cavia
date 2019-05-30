"""
Regression experiment using CAVIA
"""
import copy
import os
import time
import warnings

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils
from regression import tasks_sine, tasks_celebA
from regression.cavia_model import CaviaModel
from regression.default_configs import get_default_config_cavia
from regression.logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config, log_interval=5000, rerun=False):
    assert config['method'] == 'cavia'

    # see if we already ran this experiment
    code_root = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir('{}/{}_result_files/'.format(code_root, config['task'])):
        os.mkdir('{}/{}_result_files/'.format(code_root, config['task']))
    path = '{}/{}_result_files/'.format(code_root, config['task']) + utils.get_path_from_config(config)

    if os.path.exists(path + '.pkl') and not rerun:
        return utils.load_obj(path)

    start_time = time.time()
    utils.set_seed(config['seed'])

    # warn if there's more keys in the configs than should be
    if not len(config.keys()) == len(get_default_config_cavia(config['task']).keys()):
        warnings.warn('It seems like additional keys were added to the default config.')
        print([k for k in config.keys() if k not in get_default_config_cavia(config['task']).keys()])

    # --- initialise everything ---

    # get the task family
    if config['task'] == 'sine':
        task_family_train = tasks_sine.RegressionTasksSinusoidal()
        task_family_valid = tasks_sine.RegressionTasksSinusoidal()
        task_family_test = tasks_sine.RegressionTasksSinusoidal()
    elif config['task'] == 'celeba':
        task_family_train = tasks_celebA.CelebADataset('train')
        task_family_valid = tasks_celebA.CelebADataset('valid')
        task_family_test = tasks_celebA.CelebADataset('test')
    else:
        raise NotImplementedError

    # initialise network
    model = CaviaModel(n_in=task_family_train.num_inputs,
                       n_out=task_family_train.num_outputs,
                       num_context_params=config['num_context_params'],
                       n_hidden=config['n_hidden'],
                       ).to(device)

    # intitialise meta-optimiser
    # (only on shared params - context parameters are *not* registered parameters of the model)
    meta_optimiser = optim.Adam(model.parameters(), config['lr_meta'])

    # initialise loggers
    logger = Logger()
    logger.best_valid_model = copy.deepcopy(model)

    # --- main training loop ---

    for i_iter in range(config['n_iter']):

        # initialise meta-gradient
        meta_gradient = [0 for _ in range(len(model.state_dict()))]

        # sample tasks
        target_functions = task_family_train.sample_tasks(config['tasks_per_metaupdate'])

        # --- inner loop ---

        for t in range(config['tasks_per_metaupdate']):

            # reset private network weights
            model.reset_context_params()

            # get data for current task
            train_inputs = task_family_train.sample_inputs(config['k_meta_train'], config['order_pixels'])

            for _ in range(config['num_inner_updates']):
                # forward through model
                train_outpus = model(train_inputs)

                # get targets
                train_targets = target_functions[t](train_inputs)

                # ------------ update on current task ------------

                # compute loss for current task
                task_loss = F.mse_loss(train_outpus, train_targets)

                # compute gradient wrt context params
                task_gradients = \
                    torch.autograd.grad(task_loss, model.context_params, create_graph=not config['first_order'])[0]

                # update context params (this will set up the computation graph correctly)
                model.context_params = model.context_params - config['lr_inner'] * task_gradients

            # ------------ compute meta-gradient on test loss of current task ------------

            # get test data
            test_inputs = task_family_train.sample_inputs(config['k_meta_test'], config['order_pixels'])

            # get outputs after update
            test_outputs = model(test_inputs)

            # get the correct targets
            test_targets = target_functions[t](test_inputs)

            # compute loss after updating context (will backprop through inner loop)
            loss_meta = F.mse_loss(test_outputs, test_targets)

            # compute gradient + save for current task
            task_grad = torch.autograd.grad(loss_meta, model.parameters())

            for i in range(len(task_grad)):
                # clip the gradient
                meta_gradient[i] += task_grad[i].detach().clamp_(-10, 10)

        # ------------ meta update ------------

        # assign meta-gradient
        for i, param in enumerate(model.parameters()):
            param.grad = meta_gradient[i] / config['tasks_per_metaupdate']

        # do update step on shared model
        meta_optimiser.step()

        # reset context params
        model.reset_context_params()

        # ------------ logging ------------

        if i_iter % log_interval == 0:

            # evaluate on training set
            loss_mean, loss_conf = eval_cavia(config, copy.deepcopy(model), task_family=task_family_train,
                                              num_updates=config['num_inner_updates'])
            logger.train_loss.append(loss_mean)
            logger.train_conf.append(loss_conf)

            # evaluate on test set
            loss_mean, loss_conf = eval_cavia(config, copy.deepcopy(model), task_family=task_family_valid,
                                              num_updates=config['num_inner_updates'])
            logger.valid_loss.append(loss_mean)
            logger.valid_conf.append(loss_conf)

            # evaluate on validation set
            loss_mean, loss_conf = eval_cavia(config, copy.deepcopy(model), task_family=task_family_test,
                                              num_updates=config['num_inner_updates'])
            logger.test_loss.append(loss_mean)
            logger.test_conf.append(loss_conf)

            # save logging results
            utils.save_obj(logger, path)

            # save best model
            if logger.valid_loss[-1] == np.min(logger.valid_loss):
                print('saving best model at iter', i_iter)
                logger.best_valid_model = copy.deepcopy(model)

            # visualise results
            if config['task'] == 'celeba':
                tasks_celebA.visualise(task_family_train, task_family_test, copy.deepcopy(logger.best_valid_model),
                                       config, i_iter)

            # print current results
            logger.print_info(i_iter, start_time)
            start_time = time.time()

    return logger


def eval_cavia(config, model, task_family, num_updates, n_tasks=100, return_gradnorm=False):
    # get the task family
    input_range = task_family.get_input_range()

    # logging
    losses = []
    gradnorms = []

    # --- inner loop ---

    for t in range(n_tasks):

        # sample a task
        target_function = task_family.sample_task()

        # reset context parameters
        model.reset_context_params()

        # get data for current task
        curr_inputs = task_family.sample_inputs(config['k_shot_eval'], config['order_pixels'])
        curr_targets = target_function(curr_inputs)

        # ------------ update on current task ------------

        for _ in range(1, num_updates + 1):

            # forward pass
            curr_outputs = model(curr_inputs)

            # compute loss for current task
            task_loss = F.mse_loss(curr_outputs, curr_targets)

            # compute gradient wrt context params
            task_gradients = \
                torch.autograd.grad(task_loss, model.context_params, create_graph=not config['first_order'])[0]

            # update context params
            if config['first_order']:
                model.context_params = model.context_params - config['lr_inner'] * task_gradients.detach()
            else:
                model.context_params = model.context_params - config['lr_inner'] * task_gradients

            # keep track of gradient norms
            gradnorms.append(task_gradients[0].norm().item())

        # ------------ logging ------------

        # compute true loss on entire input range
        model.eval()
        losses.append(F.mse_loss(model(input_range), target_function(input_range)).detach().item())
        model.train()

    losses_mean = np.mean(losses)
    losses_conf = st.t.interval(0.95, len(losses) - 1, loc=losses_mean, scale=st.sem(losses))
    if not return_gradnorm:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean))
    else:
        return losses_mean, np.mean(np.abs(losses_conf - losses_mean)), np.mean(gradnorms)


if __name__ == '__main__':

    config = get_default_config_cavia(task='sine')
    # config = get_default_config_cavia(task='celeba')

    logger = run(config, log_interval=100, rerun=True)
