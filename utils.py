#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
from tqdm import trange
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import os


def name_formula(name):
    name_dict = {'cocs': 'Coauthor CS', 'cophy': 'Coauthor Physics', 'pubmed': 'Pubmed', 'film': 'Actor',
                 'cora_full': 'Cora Full', 'squirrel_filtered': 'Squirrel Filtered',
                 'chameleon_filtered': 'Chameleon Filtered', 'penn94': 'penn94', 'computers': 'Computers',
                 'photo': 'Photo'}
    return name_dict[name]

# bounded with cal_nei_index
def cal_hn(nei_dict, y, device, thres=0.5, soft=False):
    hn = np.empty(len(y), dtype=float)
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        labels = labels[labels == y[i]]
        if len(neigh):
            hn[i] = len(labels) / len(neigh)
        else:
            hn[i] = 1

    if soft:
        return hn
    mask = np.where(hn <= thres, 1., 0.)
    return torch.from_numpy(mask).float().to(device)


def cal_h(nei_dict, y):
    cc = np.empty(y.shape[0])
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        a = torch.sum(labels, dim=0) / len(labels)
        a = a.numpy()
        cc[i] = sum(np.log2(a, where=a != 0) * a * (-1))

    return cc
    # low_cc: 1 ; high_cc: 0


def cal_mode(nei_dict, y, device, thres=2., use_tensor=True, soft=False):
    cc = np.empty(y.shape[0])
    y = torch.argmax(y,dim=-1)
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        #if len(labels):
            #cc[i] = len(labels) / torch.max(torch.sum(labels, dim=0)).item()
        cc[i] =  torch.argmax(torch.sum(labels, dim=0),dim=-1)==y[i]
        #else:
        #    cc[i] = 1.0

    if soft:
        return torch.from_numpy(cc).float().to(device)
    # low_cc: 1 ; high_cc: 0
    # mask = np.where(cc <= thres, 1., 0.)
    mask=cc
    if use_tensor:
        return torch.from_numpy(mask).float().to(device)
    else:
        return mask


def cal_cc(nei_dict, y, device, thres=2., use_tensor=True):
    cc = np.empty(y.shape[0])
    for i, neigh in nei_dict.items():
        labels = torch.index_select(y, 0, neigh)
        if len(labels):
            # cc[i] = len(labels) / torch.mean(torch.sum(labels, dim=0)).item()
            cc[i] = len(labels) / torch.max(torch.sum(labels, dim=0)).item()
        else:
            cc[i] = 1.0
    #if soft:
      #  return cc
    # low_cc: 1 ; high_cc: 0
    mask = np.where(cc <= thres, 1., 0.)
    if use_tensor:
        return torch.from_numpy(mask).float().to(device)
    else:
        return mask


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    data.test_mask = index_to_mask(
        rest_index[val_lb:], size=data.num_nodes)

    return data
def gpr_splits(data, num_nodes, num_classes, percls_trn=20, val_lb=500):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=num_nodes)
    data.val_mask = index_to_mask(rest_index[:val_lb], size=num_nodes)
    data.test_mask = index_to_mask(
        rest_index[val_lb:], size=num_nodes)

    return data


