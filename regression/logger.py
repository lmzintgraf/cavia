import time

import numpy as np


class Logger:

    def __init__(self):
        self.train_loss = []
        self.train_conf = []

        self.valid_loss = []
        self.valid_conf = []

        self.test_loss = []
        self.test_conf = []

        self.best_valid_model = None

    def print_info(self, iter_idx, start_time):
        print(
            'Iter {:<4} - time: {:<5} - [train] loss: {:<6} (+/-{:<6}) - [valid] loss: {:<6} (+/-{:<6}) - [test] loss: {:<6} (+/-{:<6})'.format(
                iter_idx,
                int(time.time() - start_time),
                np.round(self.train_loss[-1], 4),
                np.round(self.train_conf[-1], 4),
                np.round(self.valid_loss[-1], 4),
                np.round(self.valid_conf[-1], 4),
                np.round(self.test_loss[-1], 4),
                np.round(self.test_conf[-1], 4),
            )
        )
