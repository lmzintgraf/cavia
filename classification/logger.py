import copy
import time

import numpy as np
import torch
from torch.nn import functional as F


class Logger:
    def __init__(self, log_interval, args, verbose=True):

        self.log_interval = log_interval
        self.args = args
        self.verbose = verbose

        # highest accuracy observed on training and validation set, and on average
        self.best_train_accuracy = 0
        self.best_valid_accuracy = 0
        self.best_train_valid_accuracy = 0

        # accuracies (train/valid) we see for the best models
        self.best_model_train_stats = [0, 0]
        self.best_model_valid_stats = [0, 0]
        self.best_model_train_valid_stats = [0, 0]

        # print some infos
        if verbose:
            print(
                'n-way: ', args.n_way, 'k-shot', args.k_shot,
                'lr[in]:', args.lr_inner, ', [out]:', args.lr_meta,
                ', grad[in]:', args.num_grad_steps_inner, ', [out]:', args.num_grad_steps_eval,
                'batchs:', args.tasks_per_metaupdate,
            )

        # initialise dictionary to keep track of accuracies/losses
        # - for training tasks
        self.training_stats = {
            #
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
            #
            'train_loss_pre_update': [],
            'test_loss_pre_update': [],
            'train_loss_post_update': [],
            'test_loss_post_update': [],
        }
        # - for validation tasks
        self.validation_stats = {
            #
            'train_accuracy_pre_update': [],
            'test_accuracy_pre_update': [],
            'train_accuracy_post_update': [],
            'test_accuracy_post_update': [],
            #
            'train_loss_pre_update': [],
            'test_loss_pre_update': [],
            'train_loss_post_update': [],
            'test_loss_post_update': [],
        }

        # keep track of how long the experiment takes
        self.start_time = time.time()

    def prepare_inner_loop(self, iter_counter, mode='train'):
        """
        Called before iterating over tasks in the inner loop
        :return:
        """
        if iter_counter % self.log_interval == 0:

            if mode == 'train':
                for key in self.training_stats.keys():
                    self.training_stats[key].append([])
            elif mode == 'valid':
                for key in self.validation_stats.keys():
                    self.validation_stats[key].append([])
            else:
                raise NotImplementedError()

    def log_pre_update(self, iter_counter, support_x, support_y, query_x, query_y, model, mode='train'):

        if iter_counter % self.log_interval == 0:
            if mode == 'train':
                self.training_stats['train_accuracy_pre_update'][-1].append(
                    self.get_accuracy(support_x, support_y, model))
                self.training_stats['test_accuracy_pre_update'][-1].append(self.get_accuracy(query_x, query_y, model))
                self.training_stats['train_loss_pre_update'][-1].append(self.get_loss(support_x, support_y, model))
                self.training_stats['test_loss_pre_update'][-1].append(self.get_loss(query_x, query_y, model))
            elif mode == 'valid':
                self.validation_stats['train_accuracy_pre_update'][-1].append(
                    self.get_accuracy(support_x, support_y, model))
                self.validation_stats['test_accuracy_pre_update'][-1].append(self.get_accuracy(query_x, query_y, model))
                self.validation_stats['train_loss_pre_update'][-1].append(self.get_loss(support_x, support_y, model))
                self.validation_stats['test_loss_pre_update'][-1].append(self.get_loss(query_x, query_y, model))
            else:
                raise NotImplementedError()

    def log_post_update(self, iter_counter, support_x, support_y, query_x, query_y, model, mode='train'):

        if iter_counter % self.log_interval == 0:
            if mode == 'train':
                self.training_stats['train_accuracy_post_update'][-1].append(
                    self.get_accuracy(support_x, support_y, model))
                self.training_stats['test_accuracy_post_update'][-1].append(self.get_accuracy(query_x, query_y, model))
                self.training_stats['train_loss_post_update'][-1].append(self.get_loss(support_x, support_y, model))
                self.training_stats['test_loss_post_update'][-1].append(self.get_loss(query_x, query_y, model))
            elif mode == 'valid':
                self.validation_stats['train_accuracy_post_update'][-1].append(
                    self.get_accuracy(support_x, support_y, model))
                self.validation_stats['test_accuracy_post_update'][-1].append(
                    self.get_accuracy(query_x, query_y, model))
                self.validation_stats['train_loss_post_update'][-1].append(self.get_loss(support_x, support_y, model))
                self.validation_stats['test_loss_post_update'][-1].append(self.get_loss(query_x, query_y, model))
            else:
                raise NotImplementedError()

    def summarise_inner_loop(self, mode):
        if mode == 'train':
            for key in self.training_stats.keys():
                self.training_stats[key][-1] = np.mean(self.training_stats[key][-1])
        if mode == 'valid':
            for key in self.validation_stats.keys():
                self.validation_stats[key][-1] = np.mean(self.validation_stats[key][-1])

    def update_best_model(self, model, save_path):

        # get the current training and validation accuracy
        train_acc = self.training_stats['test_accuracy_post_update'][-1]
        valid_acc = self.validation_stats['test_accuracy_post_update'][-1]
        train_valid_acc = 0.5 * (train_acc + valid_acc)

        if train_acc > self.best_train_accuracy:

            self.best_train_accuracy = train_acc
            # save the model with the highest training accuracy so far
            self.best_model_train = copy.copy(model)
            # log what the corresponding accuracy on training and validation set are
            self.best_model_train_stats = [self.training_stats['test_accuracy_post_update'][-1],
                                           self.validation_stats['test_accuracy_post_update'][-1]]

            if save_path is not None:
                np.save(save_path + '_best_train', self.best_model_train_stats)
                save_model = self.best_model_train
                if self.args.device == 'cuda:0':
                    save_model = copy.deepcopy(self.best_model_train).to(torch.device('cpu'))
                torch.save(save_model, save_path + '_best_train')

        if valid_acc > self.best_valid_accuracy:

            self.best_valid_accuracy = valid_acc
            self.best_model_valid = copy.copy(model)
            self.best_model_valid_stats = [self.training_stats['test_accuracy_post_update'][-1],
                                           self.validation_stats['test_accuracy_post_update'][-1]]

            if save_path is not None:
                np.save(save_path + '_best_valid', self.best_model_valid_stats)
                # save model to CPU
                save_model = self.best_model_valid
                if self.args.device == 'cuda:0':
                    save_model = copy.deepcopy(self.best_model_valid).to(torch.device('cpu'))
                torch.save(save_model, save_path + '_best_valid')

        if train_valid_acc > self.best_train_valid_accuracy:

            self.best_train_valid_accuracy = train_valid_acc
            self.best_model_train_valid = copy.copy(model)
            self.best_model_train_valid_stats = [self.training_stats['test_accuracy_post_update'][-1],
                                                 self.validation_stats['test_accuracy_post_update'][-1]]

            if save_path is not None:
                np.save(save_path + '_best_train_valid', self.best_model_train_valid_stats)
                # save model to CPU
                save_model = self.best_model_train_valid
                if self.args.device == 'cuda:0':
                    save_model = copy.deepcopy(self.best_model_train_valid).to(torch.device('cpu'))
                torch.save(save_model, save_path + '_best_train_valid')

    def print(self, iter_counter, grad_inner, grad_meta):
        if self.verbose and (iter_counter % self.log_interval == 0):
            self.print_logs(iter_counter, grad_inner, grad_meta)

    def print_header(self):
        if self.verbose:
            print(
                '||------||------------------------ TRAINING -------------------------||------------------------ VALIDATION -----------------------||-------------------||-------||')
            print(
                '||------||----------- LOSS ------------|------------ ACC ------------||----------- LOSS ------------|------------ ACC ------------||------ GRAD -------||-------||')
            print(
                '|| iter ||    train     |     test     |    train     |     test     ||    train     |     test     |    train     |     test     ||  inner  |  meta   || time  ||')
            print(
                '||------||--------------|--------------|--------------|--------------||--------------|--------------|--------------|--------------||---------|---------||-------||')

    def print_logs(self, iter_counter, grad_inner, grad_meta):
        if self.verbose:
            print(
                '||{:5} || {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} || {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} | {:<5}->{:<5} || {:<7} | {:<7} || {:<5} ||'.format(
                    iter_counter,
                    # meta-train, task-train
                    np.round(self.training_stats['train_loss_pre_update'][-1], 3),
                    np.round(self.training_stats['train_loss_post_update'][-1], 3),
                    np.round(self.training_stats['test_loss_pre_update'][-1], 3),
                    np.round(self.training_stats['test_loss_post_update'][-1], 3),
                    # meta-train, task-test
                    np.round(self.training_stats['train_accuracy_pre_update'][-1], 3),
                    np.round(self.training_stats['train_accuracy_post_update'][-1], 3),
                    np.round(self.training_stats['test_accuracy_pre_update'][-1], 3),
                    np.round(self.training_stats['test_accuracy_post_update'][-1], 3),
                    # meta-valid, task-train
                    np.round(self.validation_stats['train_loss_pre_update'][-1], 3),
                    np.round(self.validation_stats['train_loss_post_update'][-1], 3),
                    np.round(self.validation_stats['test_loss_pre_update'][-1], 3),
                    np.round(self.validation_stats['test_loss_post_update'][-1], 3),
                    # meta-valid, task-test
                    np.round(self.validation_stats['train_accuracy_pre_update'][-1], 3),
                    np.round(self.validation_stats['train_accuracy_post_update'][-1], 3),
                    np.round(self.validation_stats['test_accuracy_pre_update'][-1], 3),
                    np.round(self.validation_stats['test_accuracy_post_update'][-1], 3),
                    # gradients
                    np.round(grad_inner[0].abs().mean().item(), 3),
                    np.round(grad_meta[0].abs().mean().item(), 3),
                    # time
                    np.round((time.time() - self.start_time) / 60, 2)
                ))

    def get_accuracy(self, x, y, model):
        predictions = model(x)
        num_correct = torch.argmax(F.softmax(predictions, dim=1), 1).eq(y).sum().item()
        return num_correct / len(y)

    def get_loss(self, x, y, model):
        predictions = model(x)
        return F.cross_entropy(predictions, y).item()
