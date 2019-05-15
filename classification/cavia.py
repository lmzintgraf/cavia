import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from classification import utils, configs_default
from classification.dataset_miniimagenet import MiniImagenet
from classification.models import Net, CondConvNet
from classification.train_logs import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config, num_workers=1, log_interval=100, verbose=True, save_path=None):
    utils.seed(config['seed'])

    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model

    if (config['model'] == 'cnn') and config['film'] and isinstance(config['num_context_params'], list):
        model = CondConvNet(num_context_params=config['num_context_params'],
                            num_classes=config['n_way'],
                            num_filters=config['num_filters'],
                            max_pool=config['max_pool'],
                            num_film_hidden_layers=config['num_film_hidden_layers'],
                            imsize=config['imsize'],
                            batchnorm_at_films=config['batchnorm_at_films'],
                            initialisation=config['initialisation']
                            )
    else:
        model = Net(num_context_params=config['num_context_params'], num_classes=config['n_way'])
    model.train()

    # set up meta-optimiser for model parameters
    if isinstance(config['lr_meta'], float):
        meta_optimiser = torch.optim.Adam(model.parameters(), config['lr_meta'])
    elif config['lr_meta'] == 'decay':
        meta_optimiser = torch.optim.Adam(model.parameters(), 0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, 0.9)
    else:
        raise NotImplementedError('invalid learning rate')

    # initialise logger
    logger = Logger(log_interval, config, verbose=verbose)

    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]

    iter_counter = 0
    while iter_counter < config['n_iter']:

        # batchsz here means total episode number
        dataset_train = MiniImagenet(mode='train',
                                     n_way=config['n_way'], k_shot=config['k_shot'],
                                     k_query=config['k_query'],
                                     batchsz=10000, imsize=config['imsize'])
        # fetch meta_batchsz num of episode each time
        dataloader_train = DataLoader(dataset_train, config['tasks_per_metaupdate'], shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=False)

        # initialise dataloader
        dataset_valid = MiniImagenet(mode='val',
                                     n_way=config['n_way'],
                                     k_shot=config['k_shot'], k_query=config['k_query'],
                                     batchsz=500, imsize=config['imsize'])
        dataloader_valid = DataLoader(dataset_valid, batch_size=num_workers, shuffle=True, num_workers=num_workers,
                                      pin_memory=True)

        logger.print_header()

        for step, batch in enumerate(dataloader_train):

            if config['lr_meta'] == 'decay':
                scheduler.step()

            support_x = batch[0].to(device)
            support_y = batch[1].to(device)
            query_x = batch[2].to(device)
            query_y = batch[3].to(device)

            # skip batch if we don't have enough tasks in the current batch (might happen in last batch)
            if support_x.shape[0] != config['tasks_per_metaupdate']:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)

            logger.prepare_inner_loop(iter_counter)

            for inner_batch_idx in range(config['tasks_per_metaupdate']):

                # reset context parameters
                model.reset_context_params()

                # -------------- inner update --------------

                logger.log_pre_update(iter_counter, support_x[inner_batch_idx], support_y[inner_batch_idx],
                                      query_x[inner_batch_idx], query_y[inner_batch_idx], model)

                for _ in range(config['num_grad_steps_inner']):
                    # forward train data through net
                    pred_train = model(support_x[inner_batch_idx])

                    # compute loss
                    task_loss_train = F.cross_entropy(pred_train, support_y[inner_batch_idx])

                    # compute gradient for context parameters
                    task_grad_train = torch.autograd.grad(task_loss_train,
                                                          model.context_params,
                                                          create_graph=True,
                                                          retain_graph=True)[0]

                    # set context parameters to their updated values
                    model.context_params = model.context_params - config['lr_inner'] * task_grad_train

                # -------------- get meta gradient --------------

                # forward test data through updated net
                pred_test = model(query_x[inner_batch_idx])

                # compute loss on test data
                task_loss_test = F.cross_entropy(pred_test, query_y[inner_batch_idx])

                # compute gradient for shared parameters
                task_grad_test = torch.autograd.grad(task_loss_test, model.parameters())

                # add to meta-gradient
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()

                # ------------------------------------------------

                logger.log_post_update(iter_counter, support_x[inner_batch_idx], support_y[inner_batch_idx],
                                       query_x[inner_batch_idx], query_y[inner_batch_idx], model)

            # reset context parameters
            model.reset_context_params()

            # summarise inner loop and get validation performance
            logger.summarise_inner_loop(mode='train')

            if iter_counter % log_interval == 0:
                # evaluate how good the current model is (*before* updating so we can compare better)
                evaluate(iter_counter, config, model, logger, dataloader_valid, save_path)
                if save_path is not None:
                    np.save(save_path, [logger.training_stats, logger.validation_stats])
                    # save model to CPU
                    save_model = model
                    if device == 'cuda:0':
                        save_model = copy.deepcopy(model).to(torch.device('cpu'))
                    torch.save(save_model, save_path)

            logger.print(iter_counter, task_grad_train, meta_grad)
            iter_counter += 1
            if iter_counter > config['n_iter']:
                break

            # -------------- meta update --------------

            meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / config['tasks_per_metaupdate']
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()

    model.reset_context_params()
    return logger, model


def evaluate(iter_counter, config, model, logger, dataloader, save_path):
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for c, batch in enumerate(dataloader):

        support_x = batch[0].to(device)
        support_y = batch[1].to(device)
        query_x = batch[2].to(device)
        query_y = batch[3].to(device)

        for inner_batch_idx in range(support_x.shape[0]):

            # reset context parameters
            model.reset_context_params()

            # -------------- inner update --------------

            logger.log_pre_update(iter_counter,
                                  support_x[inner_batch_idx], support_y[inner_batch_idx],
                                  query_x[inner_batch_idx], query_y[inner_batch_idx],
                                  model, mode='valid')

            for _ in range(config['num_grad_steps_eval']):
                # forward train data through net
                pred_train = model(support_x[inner_batch_idx])

                # compute loss
                loss_inner = F.cross_entropy(pred_train, support_y[inner_batch_idx])

                # compute gradient for context parameters
                grad_inner = torch.autograd.grad(loss_inner,
                                                 model.context_params,
                                                 create_graph=True,
                                                 retain_graph=True)[0]

                # set context parameters to their updated values
                model.context_params = model.context_params - config['lr_inner'] * grad_inner

            logger.log_post_update(iter_counter,
                                   support_x[inner_batch_idx], support_y[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   model, mode='valid')

    # reset context parameters
    model.reset_context_params()

    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(model, save_path)


if __name__ == '__main__':

    config = configs_default.get_default_configs_cavia()

    # --- settings ---

    path = os.path.join(utils.get_base_path(), 'result_files', utils.get_path_from_config(config))
    log_interval = 100

    if not os.path.exists(path + '.npy'):
        run(config, num_workers=1, log_interval=log_interval, save_path=path)

    # -------------- plot -----------------

    plt.switch_backend('agg')
    training_stats, validation_stats = np.load(path + '.npy')

    plt.figure(figsize=(10, 5))
    x_ticks = np.arange(1, log_interval * len(training_stats['train_accuracy_pre_update']), log_interval)

    # training set
    plt.subplot(1, 2, 1)
    p = plt.plot(x_ticks, training_stats['train_accuracy_pre_update'], '--', label='[train] pre-update')
    plt.plot(x_ticks, training_stats['train_accuracy_post_update'], color=p[-1].get_color(),
             label='[train] post-update')
    p = plt.plot(x_ticks, training_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, training_stats['test_accuracy_post_update'], color=p[-1].get_color(), linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.xlim([0, 60000])

    # validation set
    plt.subplot(1, 2, 2)
    p = plt.plot(x_ticks, validation_stats['train_accuracy_pre_update'], '--', label='[train] pre-update')
    plt.plot(x_ticks, validation_stats['train_accuracy_post_update'], color=p[-1].get_color(),
             label='[train] post-update')
    p = plt.plot(x_ticks, validation_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, validation_stats['test_accuracy_post_update'], color=p[-1].get_color(), linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.xlim([0, 60000])

    if config['k_shot'] == 1:
        plt.plot(x_ticks, np.zeros(x_ticks.shape) + 0.48, 'k--')
    elif config['k_shot'] == 5:
        plt.plot(x_ticks, np.zeros(x_ticks.shape) + 0.63, 'k--')

    title = 'k={}, init={}, #t={}, lr={}-{}, ' \
            'grad={}-{} phi={} e={} #f={} bn={} i={}'.format(config['k_shot'],
                                                             config['initialisation'],
                                                             config['tasks_per_metaupdate'],
                                                             config['lr_inner'],
                                                             config['lr_meta'],
                                                             config['num_grad_steps_inner'],
                                                             config['num_grad_steps_eval'],
                                                             config['num_context_params'],
                                                             config['gradient_noise'],
                                                             config['num_film_hidden_layers'],
                                                             str(config['batchnorm_at_films']),
                                                             config['n_iter'])
    plt.suptitle(title)
    plt.title(' ')
    plt.xlabel('num iter', fontsize=10)
    plt.ylabel('accuracy', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig('result_plots/{}'.format(title.replace('.', '')))
    plt.show()
