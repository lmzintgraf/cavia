import copy
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms


def ravel_index(x, y, img_size):
    x = x / img_size[1]
    y = y % img_size[1]
    return x, y


def unravel_index(x, y, img_size):
    return ((x - 1) * img_size[1] + y).long()


class CelebADataset:
    """
    Same regression task as in Garnelo et al. 2018 (Conditional Neural Processes)
    """

    def __init__(self, mode, device):

        self.device = device

        if os.path.isdir('/home/scratch/luiraf/work/data/celeba/'):
            data_root = '/home/scratch/luiraf/work/data/celeba/'
        else:
            raise FileNotFoundError('Can\'t find celebrity faces.')

        self.code_root = os.path.dirname(os.path.realpath(__file__))
        self.imgs_root = os.path.join(data_root, 'Img/img_align_celeba/')
        self.imgs_root_preprocessed = os.path.join(data_root, 'Img/img_align_celeba_preprocessed/')
        if not os.path.isdir(self.imgs_root_preprocessed):
            os.mkdir(self.imgs_root_preprocessed)
        self.data_split_file = os.path.join(data_root, 'Eval/list_eval_partition.txt')

        # input: x-y coordinate
        self.num_inputs = 2
        # output: pixel values (RGB)
        self.num_outputs = 3

        # get the labels (train/valid/test)
        train_imgs, valid_imgs, test_imgs = self.get_labels()
        if mode == 'train':
            self.image_files = train_imgs
        elif mode == 'valid':
            self.image_files = valid_imgs
        elif mode == 'test':
            self.image_files = test_imgs
        else:
            raise ValueError

        self.img_size = (32, 32, 3)
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize((self.img_size[0], self.img_size[1]), Image.LANCZOS),
                                             transforms.ToTensor(),
                                             ])

    def sample_task(self):
        """ Sampling a task means sampling an image. """
        # choose image
        img_file = np.random.choice(self.image_files)
        img = self.get_image(img_file)
        return self.get_target_function(img)

    def get_image(self, filename):
        img_path = os.path.join(self.imgs_root, filename)
        img = self.transform(img_path).float().to(self.device)
        # img = img * 2 - 1
        img = img.permute(1, 2, 0)
        return img

    def get_target_function(self, img):
        def target_function(coordinates):
            c = copy.deepcopy(coordinates)
            # de-normalise coordinates
            c[:, 0] *= self.img_size[0]
            c[:, 1] *= self.img_size[1]
            pixel_values = img[c[:, 0].long(), c[:, 1].long(), :]
            return pixel_values

        return target_function

    def sample_tasks(self, num_tasks):
        image_files = np.random.choice(self.image_files, num_tasks, replace=False)
        target_functions = []
        for i in range(num_tasks):
            img = self.get_image(image_files[i])
            target_functions.append(self.get_target_function(img))
        return target_functions

    def sample_inputs(self, batch_size, order_pixels):
        if order_pixels:
            flattened_indices = list(range(self.img_size[0] * self.img_size[1]))[:batch_size]
        else:
            flattened_indices = np.random.choice(list(range(self.img_size[0] * self.img_size[1])), batch_size,
                                                 replace=False)
        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coordinates = np.vstack((x, y)).T
        coordinates = torch.from_numpy(coordinates).float().to(self.device)
        # normalise coordinates
        coordinates[:, 0] /= self.img_size[0]
        coordinates[:, 1] /= self.img_size[1]
        return coordinates

    def get_input_range(self):
        flattened_indices = range(self.img_size[0] * self.img_size[1])
        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coordinates = np.vstack((x, y)).T
        coordinates = torch.from_numpy(coordinates).float().to(self.device)
        # normalise coordinates
        coordinates[:, 0] /= self.img_size[0]
        coordinates[:, 1] /= self.img_size[1]
        return coordinates

    def get_labels(self):
        with open(self.data_split_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            train_imgs = []
            valid_imgs = []
            test_imgs = []
            for row in csv_reader:
                if row[1] == '0':
                    train_imgs.append(row[0])
                elif row[1] == '1':
                    valid_imgs.append(row[0])
                elif row[1] == '2':
                    test_imgs.append(row[0])
        return train_imgs, valid_imgs, test_imgs

    def visualise(self, task_family_train, task_family_test, model, args, i_iter):
        plt.figure(figsize=(14, 14))

        for i, img_path in enumerate(task_family_train.image_files[:6] + task_family_test.image_files[:6]):

            # randomly pick image
            img = task_family_train.get_image(img_path)
            # get target function
            target_func = task_family_train.get_target_function(img)
            # pick data points for training
            pixel_inputs = task_family_train.sample_inputs(args.k_shot_eval, args.use_ordered_pixels)
            pixel_targets = target_func(pixel_inputs)
            # update model
            if not args.maml:
                model.reset_context_params()
                for _ in range(args.num_inner_updates):
                    pixel_pred = model(pixel_inputs)
                    loss = F.mse_loss(pixel_pred, pixel_targets)
                    grad = torch.autograd.grad(loss, model.context_params, create_graph=not args.first_order)[0]
                    model.context_params = model.context_params - args.lr_inner * grad
            else:
                for _ in range(args.num_inner_updates):
                    pixel_pred = model(pixel_inputs)
                    loss = F.mse_loss(pixel_pred, pixel_targets)
                    params = [w for w in model.weights] + [b for b in model.biases] + [model.task_context]
                    grads = torch.autograd.grad(loss, params)
                    for k in range(len(model.weights)):
                        model.weights[k] = model.weights[k] - args.lr_inner * grads[k].detach()
                    for j in range(len(model.biases)):
                        model.biases[j] = model.biases[j] - args.lr_inner * grads[k + j + 1].detach()
                    model.task_context = model.task_context - args.lr_inner * grads[k + j + 2].detach()

            # plot context
            plt.subplot(6, 6, (i % 6) * 6 + 1 + int(i > 5) * 3)
            # img = (img + 1) / 2
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

            # context
            plt.subplot(6, 6, (i % 6) * 6 + 2 + int(i > 5) * 3)
            img_copy = copy.copy(img) * 0
            # de-normalise coordinates
            pixel_inputs *= 32
            pixel_inputs = pixel_inputs.long()
            img_copy[pixel_inputs[:, 0], pixel_inputs[:, 1]] = img[pixel_inputs[:, 0], pixel_inputs[:, 1]]
            plt.imshow(img_copy)
            plt.xticks([])
            plt.yticks([])

            if i == 0:
                plt.title('TRAIN', fontsize=20)
            if i == 6:
                plt.title('TEST', fontsize=20)

            # predict
            plt.subplot(6, 6, (i % 6) * 6 + 3 + int(i > 5) * 3)
            input_range = task_family_train.get_input_range()
            img_pred = model(input_range).view(task_family_train.img_size).cpu().detach().numpy()
            # img_pred = (img_pred + 1) / 2
            img_pred[img_pred < 0] = 0
            img_pred[img_pred > 1] = 1
            plt.imshow(img_pred)
            plt.xticks([])
            plt.yticks([])

        if not os.path.isdir('{}/celeba_result_plots/'.format(self.code_root)):
            os.mkdir('{}/celeba_result_plots/'.format(self.code_root))

        plt.tight_layout()
        plt.savefig('{}/celeba_result_plots/{}_c{}_k{}_o{}_u{}_lr{}_{}'.format(self.code_root,
                                                                               int(args.maml),
                                                                               args.num_context_params,
                                                                               args.k_meta_train,
                                                                               args.use_ordered_pixels,
                                                                               args.num_inner_updates,
                                                                               int(10 * args.lr_inner),
                                                                               i_iter))
        plt.close()
