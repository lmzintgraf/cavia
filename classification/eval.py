"""
This script evaluates a saved model (will crash if nothing is saved).
"""
import os
import time

import numpy as np
import scipy.stats as st
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils
import arguments
from dataset_miniimagenet import MiniImagenet


def evaluate(args, model, logger, dataloader, mode, num_grad_steps):

    for c, batch in enumerate(dataloader):

        support_x = batch[0].to(args.device)
        support_y = batch[1].to(args.device)
        query_x = batch[2].to(args.device)
        query_y = batch[3].to(args.device)

        for inner_batch_idx in range(support_x.shape[0]):

            # reset context parameters
            model.reset_context_params()

            logger.log_pre_update(support_x[inner_batch_idx], support_y[inner_batch_idx],
                                  query_x[inner_batch_idx], query_y[inner_batch_idx],
                                  model, mode)

            for _ in range(num_grad_steps):
                # forward train data through net
                pred_train = model(support_x[inner_batch_idx])

                # compute loss
                loss_inner = F.cross_entropy(pred_train, support_y[inner_batch_idx])

                # compute gradient for context parameters
                grad_inner = torch.autograd.grad(loss_inner, model.context_params)[0]

                # set context parameters to their updated values
                model.context_params = model.context_params - args.lr_inner * grad_inner

            logger.log_post_update(support_x[inner_batch_idx], support_y[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   model, mode)

    # reset context parameters
    model.reset_context_params()


class Logger:
    def __init__(self, args):

        self.args = args

        # initialise dictionary to keep track of accuracies/losses
        self.train_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }
        self.valid_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }
        self.test_stats = {
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
        }

        # keep track of how long the experiment takes
        self.start_time = time.time()

    def log_pre_update(self, support_x, support_y, query_x, query_y, model, mode):
        if mode == 'train':
            self.train_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, model))
            self.train_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, model))
        elif mode == 'val':
            self.valid_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, model))
            self.valid_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, model))
        elif mode == 'test':
            self.test_stats['train_accuracy_pre_update'].append(self.get_accuracy(support_x, support_y, model))
            self.test_stats['test_accuracy_pre_update'].append(self.get_accuracy(query_x, query_y, model))

    def log_post_update(self, support_x, support_y, query_x, query_y, model, mode):
        if mode == 'train':
            self.train_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, model))
            self.train_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, model))
        elif mode == 'val':
            self.valid_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, model))
            self.valid_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, model))
        elif mode == 'test':
            self.test_stats['train_accuracy_post_update'].append(self.get_accuracy(support_x, support_y, model))
            self.test_stats['test_accuracy_post_update'].append(self.get_accuracy(query_x, query_y, model))

    def print_header(self):
        print(
            '||------------------------------------------------------------------------------------------------------------------------------------------------------||')
        print(
            '||------------- TRAINING ------------------------|---------------------------------------- EVALUATION --------------------------------------------------||')
        print(
            '||-----------------------------------------------|------------------------------------------------------------------------------------------------------||')
        print(
            '||-----------------|     observed performance    |          META_TRAIN         |           META_VALID         |           META_TEST                     ||')
        print(
            '||    selection    |-----------------------------|-----------------------------|------------------------------|-----------------------------------------||')
        print(
            '||    criterion    |    train     |     valid    |    train     |     test     |    train     |     test      |    train     |     test                 ||')
        print(
            '||-----------------|--------------|--------------|--------------|--------------|--------------|---------------|--------------|--------------------------||')

    def print_logs(self, selection_criterion, logged_perf=None):
        if logged_perf is None:
            logged_perf = [' ', ' ']
        else:
            logged_perf = [np.round(logged_perf[0], 3), np.round(logged_perf[1], 3)]

        avg_acc = np.mean(self.test_stats['test_accuracy_post_update'])
        conf_interval = st.t.interval(0.95,
                                      len(self.test_stats['test_accuracy_post_update']) - 1,
                                      loc=avg_acc,
                                      scale=st.sem(self.test_stats['test_accuracy_post_update']))

        print(
            '||   {:<11}   |    {:<5}     |     {:<5}    | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5}  | {:<5}->{:<5} | {:<5}->{:<5} (+/- {}) ||'.format(
                selection_criterion,
                # performance we observed during training
                logged_perf[0],
                logged_perf[1],
                # meta-valid, task-test
                np.round(np.mean(self.train_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.train_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.train_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.train_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.valid_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.valid_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.valid_stats['test_accuracy_pre_update']), 3),
                np.round(np.mean(self.valid_stats['test_accuracy_post_update']), 3),
                #
                np.round(np.mean(self.test_stats['train_accuracy_pre_update']), 3),
                np.round(np.mean(self.test_stats['train_accuracy_post_update']), 3),
                np.round(np.mean(self.test_stats['test_accuracy_pre_update']), 2),
                np.round(100 * avg_acc, 3),
                #
                np.round(100 * np.mean(np.abs(avg_acc - conf_interval)), 2),
            ))

    def get_accuracy(self, x, y, model):
        predictions = model(x)
        num_correct = torch.argmax(F.softmax(predictions, dim=1), 1).eq(y).sum().item()
        return num_correct / len(y)


if __name__ == '__main__':

    args = arguments.parse_args()
    log_interval = 100

    utils.set_seed(args.seed)

    # --- settings ---

    args.k_shot = 1
    args.lr_inner = 1.0
    args.lr_meta = 'decay'
    args.num_grad_steps_inner = 2
    args.num_grad_steps_eval = 2
    args.model = 'cnn'
    args.num_context_params = 100

    if args.k_shot == 1:
        args.tasks_per_metaupdate = 4
    else:
        args.tasks_per_metaupdate = 2

    path = os.path.join(utils.get_base_path(), 'result_files', utils.get_path_from_args(args))

    try:
        training_stats, validation_stats = np.load(path + '.npy')
    except FileNotFoundError:
        print('You need to run the experiments first and make sure the results are saved at {}'.format(path))
        raise FileNotFoundError

    # initialise logger
    logger = Logger(args)
    logger.print_header()

    for num_grad_steps in [2]:

        print('\n --- ', num_grad_steps, '--- \n')

        # initialise logger
        logger = Logger(args)

        for selection_criterion in ['valid']:

            logger = Logger(args)

            for dataset in ['train', 'val', 'test']:
                # load model and its performance during training
                model = torch.load(path + '_best_{}'.format(selection_criterion))
                best_performances = np.load(path + '_best_{}.npy'.format(selection_criterion))

                # initialise dataloader
                mini_test = MiniImagenet(mode=dataset, n_way=args.n_way,
                                         k_shot=args.k_shot, k_query=args.k_query,
                                         batchsz=500, verbose=False, imsize=args.imsize)
                db_test = DataLoader(mini_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

                # evaluate the model
                evaluate(args, model, logger, db_test, mode=dataset, num_grad_steps=num_grad_steps)

            logger.print_logs(selection_criterion, best_performances)
