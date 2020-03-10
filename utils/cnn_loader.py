import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_dls(bs , aug = True, valid_size = 0.1, **kwargs):

    aug_compose = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    
    train_ds = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True, download=True, transform = aug_compose if aug else compose)
    valid_ds = torchvision.datasets.CIFAR10(root='./data-cifar10', train=True, download=True, transform = compose)

    data_len = len(train_ds)
    indices = list(range(data_len))
    split = int(np.floor(valid_size * data_len))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_dl = DataLoader(train_ds, batch_size = bs, sampler = train_sampler)
    valid_dl = DataLoader(valid_ds, batch_size = bs, sampler = valid_sampler)

    return train_dl, valid_dl


class DataBunch():
    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl,self.valid_dl,self.c_in,self.c_out = train_dl,valid_dl,c_in,c_out

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset
        
