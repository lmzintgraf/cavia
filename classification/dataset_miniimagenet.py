"""
Taken and adapted from https://github.com/dragen1860/MAML-Pytorch
"""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- train/*.jpg
        |- test/*.jpg
        |- val/*.jpg
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, mode, batchsz, n_way, k_shot, k_query, imsize, data_path, startidx=0, verbose=True):
        """
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of query imgs per class
        :param imsize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.support_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.imsize = imsize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.data_path = data_path

        if verbose:
            print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
                mode, batchsz, n_way, k_shot, k_query, imsize))

        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                             transforms.Resize((self.imsize, self.imsize), Image.LANCZOS),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])

        # check if images are all in one folder or separated into train/val/test folders
        if os.path.exists(os.path.join(data_path, 'images')):
            self.subfolder_split = False
            self.path_images = os.path.join(data_path, 'images')  # image path
            self.path_preprocessed = os.path.join(data_path, 'images_preprocessed')  # preprocessed image path
        elif os.path.exists(os.path.join(data_path, 'train')):
            self.subfolder_split = True
            self.path_images = os.path.join(data_path, mode)
            self.path_preprocessed = os.path.join(data_path, 'images_preprocessed')
            if not os.path.exists(self.path_preprocessed):
                os.mkdir(self.path_preprocessed)
            self.path_preprocessed = os.path.join(data_path, 'images_preprocessed', mode)
            if not os.path.exists(self.path_preprocessed):
                os.mkdir(self.path_preprocessed)
        else:
            raise FileNotFoundError('Mini-Imagenet data not found. '
                                    'Please add images in one of the following folder structures:'
                                    './data/miniimagenet/images'
                                    './data/miniimagenet/{train}{test}{val}'
                                    'or specify --data_path in the arguments.'
                                    )

        csvdata = [self.loadCSV(os.path.join(data_path, mode + '.csv'))]  # csv path

        # check if we have the images
        if not os.listdir(self.path_images):
            raise FileNotFoundError('Mini-Imagenet data not found. '
                                    'Please add images in one of the following folder structures:'
                                    './data/miniimagenet/images'
                                    './data/miniimagenet/{train}{test}{val}'
                                    'or specify --data_path in the arguments.'
                                    )

        self.data = []
        self.img2label = {}
        for c in csvdata:
            for i, (k, v) in enumerate(c.items()):
                self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
                self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
            self.startidx += i + 1
        self.num_classes = len(self.data)

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self, meta_iterations):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(meta_iterations):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.num_classes, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                indices_support = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indices_query = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.extend(np.array(self.data[cls])[indices_support].tolist())  # get all image filenames
                query_x.extend(np.array(self.data[cls])[indices_query].tolist())

            # shuffle the corresponding relation between support set and query set
            support_x = np.random.permutation(support_x)
            query_x = np.random.permutation(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """

        # initialise empty tensors for the images
        support_x = torch.FloatTensor(self.support_size, 3, self.imsize, self.imsize)
        query_x = torch.FloatTensor(self.query_size, 3, self.imsize, self.imsize)

        # get the filenames and labels of the images
        filenames_support_x = [item for item in self.support_x_batch[index]]
        support_y = np.array([self.img2label[item[:9]]  # filename: n0153282900000005.jpg, first 9 chars are label
                              for item in self.support_x_batch[index]]).astype(np.int32)

        filenames_query_x = [item for item in self.query_x_batch[index]]
        query_y = np.array([self.img2label[item[:9]] for item in self.query_x_batch[index]]).astype(np.int32)

        # unique: [n-way], sorted
        unique = np.random.permutation(np.unique(support_y))
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.support_size)
        query_y_relative = np.zeros(self.query_size)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # pre-process the images and save as numpy arrays (makes the code run much faster afterwards)
        # - for the support set
        for i, filename in enumerate(filenames_support_x):
            filename_preprocessed = filename[:-4] + '_preprocessed_{}'.format(self.imsize)
            path_preprocessed = os.path.join(self.path_preprocessed, filename[:9], filename_preprocessed)
            if not os.path.exists(path_preprocessed + '.npy'):
                if not os.path.exists(os.path.join(self.path_preprocessed, filename[:9])):
                    os.mkdir(os.path.join(self.path_preprocessed, filename[:9]))
                if self.subfolder_split:
                    support_x[i] = self.transform(os.path.join(self.path_images, filename[:9], filename))
                else:
                    support_x[i] = self.transform(os.path.join(self.path_images, filename))
                np.save(path_preprocessed, support_x[i].numpy())
            else:
                support_x[i] = torch.from_numpy(np.load(path_preprocessed + '.npy'))
        # - same thing for the query set
        for i, filename in enumerate(filenames_query_x):
            filename_preprocessed = filename[:-4] + '_preprocessed_{}'.format(self.imsize)
            path_preprocessed = os.path.join(self.path_preprocessed, filename[:9], filename_preprocessed)
            if not os.path.exists(path_preprocessed + '.npy'):
                if not os.path.exists(os.path.join(self.path_preprocessed, filename[:9])):
                    os.mkdir(os.path.join(self.path_preprocessed, filename[:9]))
                if self.subfolder_split:
                    query_x[i] = self.transform(os.path.join(self.path_images, filename[:9], filename))
                else:
                    query_x[i] = self.transform(os.path.join(self.path_images, filename))
                np.save(path_preprocessed, query_x[i].numpy())
            else:
                query_x[i] = torch.from_numpy(np.load(path_preprocessed + '.npy'))

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz
