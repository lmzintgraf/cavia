import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CondConvNet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_filters,
                 max_pool,
                 num_context_params,
                 num_film_hidden_layers,
                 imsize,
                 batchnorm_at_films,
                 initialisation,
                 ):
        super(CondConvNet, self).__init__()

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.max_pool = max_pool
        self.num_context_params = num_context_params
        self.num_film_hidden_layers = num_film_hidden_layers
        self.kernel_size = 3
        self.batchnorm_at_films = batchnorm_at_films

        # -- shared network --

        stride = 1
        padding = 1
        self.num_channels = 3

        # conv-layers
        self.conv1 = nn.Conv2d(self.num_channels, self.num_filters, self.kernel_size, stride=stride,
                               padding=padding).to(device)
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(
            device)
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride, padding=padding).to(
            device)
        if not self.max_pool:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride).to(device)
        else:
            self.conv4 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=stride,
                                   padding=padding).to(device)

        # batch norm (IMPORTANT: move to GPU first, then create parameter! Otherwise it won't get registered!)
        self.bn1 = nn.BatchNorm2d(self.num_filters).to(device)
        self.bn2 = nn.BatchNorm2d(self.num_filters).to(device)
        self.bn3 = nn.BatchNorm2d(self.num_filters).to(device)
        self.bn4 = nn.BatchNorm2d(self.num_filters).to(device)

        # initialise weights for the fully connected layer
        if imsize == 84:
            if len(num_context_params) == 5:
                self.fc1 = nn.Linear(5 * 5 * self.num_filters + num_context_params[4], self.num_classes).to(device)
            elif len(num_context_params) == 6:
                self.fc1 = nn.Linear(5 * 5 * self.num_filters + num_context_params[4], 256).to(device)
                self.fc2 = nn.Linear(256 + num_context_params[5], self.num_classes).to(device)
        elif imsize == 28:
            self.fc1 = nn.Linear(self.num_filters + num_context_params[4], self.num_classes).to(device)
        else:
            raise NotImplementedError('Cannot handle image size.')

        # -- additions to enable context parameters at convolutional layers --

        # for each layer where we have context parameters, initialise a FiLM layer
        if num_context_params[0] != 0:
            self.film1 = nn.Linear(self.num_context_params[0], self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film11 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if num_context_params[1] != 0:
            self.film2 = nn.Linear(self.num_context_params[1], self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film22 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if num_context_params[2] != 0:
            self.film3 = nn.Linear(self.num_context_params[2], self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film33 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)
        if num_context_params[3] != 0:
            self.film4 = nn.Linear(self.num_context_params[3], self.num_filters * 2).to(device)
            if self.num_film_hidden_layers == 1:
                self.film44 = nn.Linear(self.num_filters * 2, self.num_filters * 2).to(device)

        # parameter initialisation (if different than standard pytorch one)
        if initialisation != 'standard':
            self.init_params(initialisation)

        # initialise context parameters
        self.context_params = torch.zeros(size=[np.sum(self.num_context_params)], requires_grad=True).to(device)

    def init_params(self, initialisation):

        # convolutional weights

        if initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu', self.conv1.weight))
            torch.nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu', self.conv2.weight))
            torch.nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('relu', self.conv3.weight))
            torch.nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('relu', self.conv4.weight))
        elif initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')

        # convolutional bias

        self.conv1.bias.data.fill_(0)
        self.conv2.bias.data.fill_(0)
        self.conv3.bias.data.fill_(0)
        self.conv4.bias.data.fill_(0)

        # fully connected weights at the end

        if initialisation == 'xavier':
            if len(self.num_context_params) == 5:
                torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear', self.fc1.weight))
            elif len(self.num_context_params) == 6:
                torch.nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu', self.fc1.weight))
                torch.nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('linear', self.fc2.weight))
        elif initialisation == 'kaiming':
            if len(self.num_context_params) == 5:
                torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='linear')
            elif len(self.num_context_params) == 6:
                torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
                torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='linear')

        # fully connected bias

        self.fc1.bias.data.fill_(0)
        if len(self.num_context_params) == 6:
            self.fc2.bias.data.fill_(0)

        # FiLM layer weights

        if self.num_context_params[0] != 0 and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film1.weight, gain=nn.init.calculate_gain('linear', self.film1.weight))
        elif self.num_context_params[0] != 0 and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film1.weight, nonlinearity='linear')

        if self.num_context_params[1] != 0 and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film2.weight, gain=nn.init.calculate_gain('linear', self.film2.weight))
        elif self.num_context_params[1] != 0 and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film2.weight, nonlinearity='linear')

        if self.num_context_params[2] != 0 and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film3.weight, gain=nn.init.calculate_gain('linear', self.film3.weight))
        elif self.num_context_params[2] != 0 and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film3.weight, nonlinearity='linear')

        if self.num_context_params[3] != 0 and initialisation == 'xavier':
            torch.nn.init.xavier_uniform_(self.film4.weight, gain=nn.init.calculate_gain('linear', self.film4.weight))
        elif self.num_context_params[3] != 0 and initialisation == 'kaiming':
            torch.nn.init.kaiming_uniform_(self.film4.weight, nonlinearity='linear')

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True

    def forward(self, x):

        # pass through convolutional layer
        h1 = self.conv1(x)
        # batchnorm
        h1 = self.bn1(h1)
        # do max-pooling (for imagenet)
        if self.max_pool:
            h1 = F.max_pool2d(h1, kernel_size=2)
        # if we have context parameters, adjust conv output using FiLM variables
        if self.num_context_params[0] != 0:
            # FiLM it: forward through film layer to get scale and shift parameter
            film1 = self.film1(self.context_params[:self.num_context_params[0]])
            if self.num_film_hidden_layers == 1:
                film1 = self.film11(F.relu(film1))
            gamma1 = film1[:self.num_filters].view(1, -1, 1, 1)
            beta1 = film1[self.num_filters:].view(1, -1, 1, 1)
            # transform feature map
            h1 = gamma1 * h1 + beta1
        # pass through ReLu activation function
        h1 = F.relu(h1)

        h2 = self.conv2(h1)
        h2 = self.bn2(h2)
        if self.max_pool:
            h2 = F.max_pool2d(h2, kernel_size=2)
        if self.num_context_params[1] != 0:
            film2 = self.film2(self.context_params[self.num_context_params[0]:np.sum(self.num_context_params[:2])])
            if self.num_film_hidden_layers == 1:
                film2 = self.film22(F.relu(film2))
            gamma2 = film2[:self.num_filters].view(1, -1, 1, 1)
            beta2 = film2[self.num_filters:].view(1, -1, 1, 1)
            h2 = gamma2 * h2 + beta2
        h2 = F.relu(h2)

        h3 = self.conv3(h2)
        h3 = self.bn3(h3)
        if self.max_pool:
            h3 = F.max_pool2d(h3, kernel_size=2)
        if self.num_context_params[2] != 0:
            film3 = self.film3(
                self.context_params[np.sum(self.num_context_params[:2]):np.sum(self.num_context_params[:3])])
            if self.num_film_hidden_layers == 1:
                film3 = self.film33(F.relu(film3))
            gamma3 = film3[:self.num_filters].view(1, -1, 1, 1)
            beta3 = film3[self.num_filters:].view(1, -1, 1, 1)
            h3 = gamma3 * h3 + beta3
        h3 = F.relu(h3)

        h4 = self.conv4(h3)
        h4 = self.bn4(h4)
        if self.max_pool:
            h4 = F.max_pool2d(h4, kernel_size=2)
        if self.num_context_params[3] != 0:
            film4 = self.film4(
                self.context_params[np.sum(self.num_context_params[:3]):np.sum(self.num_context_params[:4])])
            if self.num_film_hidden_layers == 1:
                film4 = self.film44(F.relu(film4))
            gamma4 = film4[:self.num_filters].view(1, -1, 1, 1)
            beta4 = film4[self.num_filters:].view(1, -1, 1, 1)
            h4 = gamma4 * h4 + beta4
        h4 = F.relu(h4)

        # flatten
        h4 = h4.view(h4.size(0), -1)

        if self.num_context_params[4] != 0:
            h4 = torch.cat((h4, self.context_params[
                                np.sum(self.num_context_params[:4]):np.sum(self.num_context_params[:5])].expand(
                h4.size(0), -1)), dim=1)

        y = self.fc1(h4)

        if len(self.num_context_params) == 6:
            if self.num_context_params[5] != 0:
                y = torch.cat((y, self.context_params[
                                  np.sum(self.num_context_params[:5]):np.sum(self.num_context_params[:6])].expand(
                    y.size(0), -1)), dim=1)
            y = F.relu(y)
            y = self.fc2(y)

        return y


class Net(nn.Module):
    def __init__(self,
                 num_context_params,
                 num_feats,
                 num_classes
                 ):
        super(Net, self).__init__()

        self.num_context_params = num_context_params
        self.num_classes = num_classes

        # initialise weights for the fully connected layer
        self.fc1 = nn.Linear(num_feats + self.num_context_params, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

        # initialise context parameters
        self.context_params = torch.zeros(size=[self.num_context_params], requires_grad=True, device=device)

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True

    def forward(self, x):
        x = torch.cat((x, self.context_params.expand(x.size(0), -1)), dim=1)

        # pass through layers
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        y = self.fc4(h3)

        return y
