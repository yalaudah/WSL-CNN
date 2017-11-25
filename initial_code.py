#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:40:50 2017

@author: yazeed
"""

# library
# standard library
import os

import matplotlib.pyplot as plt

# third-party library
import torch
import torch.nn as nn
from torch import np # Torch wrapper for Numpy
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data_loader import get_train_valid_loader,get_test_loader

#%% Arguments (later) and parameters (for now)

data_dir='/home/yazeed/Documents/datasets/seismic-2000/' # data path
batch_size = 32
n_threads = 1 # number of workers
use_gpu = 0


if use_gpu:
    pin_memory =True


#%% Setup Dataset and DataLoaders: 

train_loader, valid_loader = get_train_valid_loader(data_dir,
                           batch_size,
                           augment=1,
                           random_seed=1,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=n_threads,
                           pin_memory=pin_memory)
    
'''
the get_train_valid_loader above is a high level wrapper for the code below. It 
allows random shuffling, doing train/val splits, and data augmentation.
'''
# create data_loader: 
#traindir = os.path.join(data_dir, 'train')
#valdir = os.path.join(data_dir, 'val')
#train = datasets.ImageFolder(traindir, transform)
#val = datasets.ImageFolder(valdir, transform)
#train_loader = torch.utils.data.DataLoader(
#    train, batch_size=batch_size, shuffle=True, num_workers=n_threads)


#%% Setup Model
#
#Net = models.vgg16(pretrained=True) # Q: how can i later delete the last 1-2 layers and use my own fully connected layer?
## define Loss Function and Optimizer
#criterion = nn.CrossEntropyLoss().cuda()
#optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum)

# SAMPLE NETWORK -- FIND A WAY TO PRETRAIN IT!
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0,dilation=0) # TODO: try with dilation (i.e. atrous convolution)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0,dilation=0) # Try with padding
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

model = Net() # On CPU
# model = Net().cuda() # On GPU



