#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:40:50 2017

@author: yazeed
"""

# standard library
#import os

#import matplotlib.pyplot as plt

# PyTorch:
import torch
import torch.nn as nn
#from torch import np # Torch wrapper for Numpy
from torch.autograd import Variable
#import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
#import torchvision.models as models
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms

# Logging:
from tensorboard import SummaryWriter

# Other: 
from data_loader import get_train_valid_loader,get_test_loader

#%% Arguments (later) and parameters (for now)

data_dir='/home/yazeed/Documents/datasets/seismic-2000/' # data path
batch_size = 20
n_threads = 1 # number of workers
use_gpu = 1
num_epochs = 100

if use_gpu:
    pin_memory = True
else:
    pin_memory = False

writer = SummaryWriter()

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
        # TODO: chnage the 3 below to 1 once that is fixed in the data.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,dilation=1) # TODO: try with dilation (i.e. atrous convolution)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1,dilation=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1,dilation=1) 
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18432, 32) # 12800
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        print(x.size())
        x = x.view(x.size(0), -1) # Flatten layer
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        print(x.size())
        return F.sigmoid(x)

if use_gpu:
    model = Net().cuda() # On GPU
else:
    model = Net() # On CPU    
    
#%% Defining the training function: 
    
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

global idx 
idx = 0

def train(epoch):
    # this only needs to be created once -- then reused:
    target_onehot = torch.FloatTensor(train_loader.batch_size,len(train_loader.dataset.classes)).zero_() 
    if use_gpu:
        target_onehot = target_onehot.cuda()
        
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        
        data, target = Variable(data), Variable(target)
        
        # Convert target to one-hot format: --------
        index = torch.unsqueeze(target.data,1)
        target_onehot.zero_()
        target_onehot.scatter_(1,index,1)
        # FOR SOME REASON BINARY CROSS ENTROPY WANTS TARGET AS FLOAT: 
        # ------------------------------------------
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.binary_cross_entropy(output, Variable(target_onehot))
        
        # ---------------------
        global idx 
        idx = idx + 1
        writer.add_scalar('training loss', loss.data[0], idx) 
        # ---------------------
        
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

#%%

def validate(epoch):
    # this only needs to be created once -- then reused:
    target_onehot = torch.FloatTensor(valid_loader.batch_size,len(valid_loader.dataset.classes)).zero_() 
    
    if use_gpu:
        target_onehot = target_onehot.cuda()
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        if use_gpu:
            data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        
        data, target = Variable(data), Variable(target)
        
        # Convert target to one-hot format: --------
        index = torch.unsqueeze(target.data,1)
        target_onehot.zero_()
        target_onehot.scatter_(1,index,1)
        # FOR SOME REASON BINARY CROSS ENTROPY WANTS TARGET AS FLOAT: 
        # ------------------------------------------
        
        output = model(data)
        
        loss = F.binary_cross_entropy(output, Variable(target_onehot))
        
        # ---------------------
        global idx 
        writer.add_scalar('validation loss', loss.data[0], idx) 
        # ---------------------
        
        if batch_idx % 10 == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                100. * batch_idx / len(valid_loader), loss.data[0]))


#%% Training: 
    
for epoch in range(1, num_epochs):
    train(epoch)
    validate(epoch)


#%%