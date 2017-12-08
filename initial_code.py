#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:40:50 2017

@author: yazeed
"""

# standard library
#import os

#import matplotlib.pyplot as plt
import numpy as np

# PyTorch:
import torch
#import torch.nn as nn
#from torch import np # Torch wrapper for Numpy
from torch.autograd import Variable
#import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models 
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from PIL import Image
# Logging:
from tensorboardX import SummaryWriter
# To run, open dir in terminal and type: tensorboard --logdir runs/exp1 to log
# different experiments in different folders  
# Other: 
from data_loader import get_train_valid_loader,get_test_loader
from helpers import models

## %% Arguments (later) and parameters (for now)

data_dir='/home/yazeed/Documents/datasets/seismic-2000/' # data path
batch_size = 20
n_threads = 1 # number of workers

num_epochs = 100

if torch.cuda.is_available():
    use_gpu = 1
    pin_memory = True
else:
    use_gpu = 0
    pin_memory = False

writer = SummaryWriter()


## ########################################################################################################
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
    

#the get_train_valid_loader above is a high level wrapper for the code below. It 
#allows random shuffling, doing train/val splits, and data augmentation.


## ########################################################################################################

#%% Setup Model
#
#Net = models.vgg16(pretrained=True) # Q: how can i later delete the last 1-2 layers and use my own fully connected layer?
## define Loss Function and Optimizer
#criterion = nn.CrossEntropyLoss().cuda()
#optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum)



Net = models.Net_temp_2


if use_gpu:
    model = Net().cuda() # On GPU
else:
    model = Net() # On CPU    
    
#%% Defining the training function: 
## ########################################################################################################

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # TODO: optimize these values

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
        
        writer.add_scalar('loss/train', loss.data[0], batch_idx) 
#        writer.add_scalars('loss',{'train_loss':loss.data[0]}, batch_idx)
        # ---------------------
        
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

#%%
## ########################################################################################################

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
        writer.add_scalar('loss/valid', loss.data[0], batch_idx) 
#        writer.add_scalars('loss',{'valid_loss':loss.data[0]}, batch_idx)
        # ---------------------
        
        if batch_idx % 10 == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                100. * batch_idx / len(valid_loader), loss.data[0]))


#%% Training: 
## ########################################################################################################


for epoch in range(1, num_epochs):
    train(epoch)
    validate(epoch)


#writer.close()

#%% Load seismic images and test: 
    
import scipy.io as sio
import matplotlib.pyplot as plt

labels = ['chaotic','fault','other','salt dome']
img  = Image.open("/home/yazeed/Documents/datasets/seismic-2000/chaotic/img_0011.png")

normalize = transforms.Normalize(mean=[0.4967, 0.4967],
                                     std=[0.1569 ,0.1569])

# To do: use the other landmass dataset, and do random crops from them -- i don't want to test on my training data.  
valid_transform = transforms.Compose([
        
            transforms.ToTensor(),
            normalize])

img = valid_transform(img)
img.unsqueeze_(0) # add 1 channel axis. Note: methods with underscore happen in place

# Add two more to make "RGB" TODO: fix this later, and make grayscale only. 
img = torch.cat((img,img,img),1)

img_var = Variable(img)
img_var = img_var.cuda() # send to GPU

op = model(img_var)
result = op.data[0].cpu().numpy()
value = np.amax(result)
index = np.argmax(result)

print('The image is: ', labels[index])

#%% TESTING ON LANDMASS-2 IMAGES: 
landmass2_dir = '/home/yazeed/Documents/datasets/landmass-2/'    
test_loader = get_test_loader(landmass2_dir,
                           batch_size,
                           shuffle=False,
                           num_workers=n_threads,
                           pin_memory=pin_memory)

# this only needs to be created once -- then reused:
target_onehot = torch.FloatTensor(test_loader.batch_size,len(test_loader.dataset.classes)).zero_() 

if use_gpu:
    target_onehot = target_onehot.cuda()

for batch_idx, (data, target) in enumerate(test_loader):
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
    writer.add_scalar('loss/test', loss.data[0], batch_idx) 
#        writer.add_scalars('loss',{'valid_loss':loss.data[0]}, idx)
    # ---------------------
    
    if batch_idx % 10 == 0:
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(test_loader.dataset),
            100. * batch_idx / len(test_loader), loss.data[0]))



#%% TRY #2:
    
normalize = transforms.Normalize(mean=[0.4967, 0.4967],
                                     std=[0.1569 ,0.1569])

test_transform = transforms.Compose([
        transforms.RandomCrop(99),
        transforms.RandomHorizontalFlip,
        transforms.ToTensor(),
        normalize
    ])

test_data = ImageFolder(root=landmass2_dir, transform=test_transform)

x,y = test_data[0] # x is the first image as PIL, y is that images class label

for i in range(len(test_data)):
    x,y = test_data[i]
