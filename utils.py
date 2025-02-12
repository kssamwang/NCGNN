#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
from tqdm import trange
import numpy as np
from torch_geometric.utils import k_hop_subgraph
import os
import torch_scatter


import logging

def start_logger():
	# logger, both log file and console
	logger = logging.getLogger('my_logger')
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	console_handler = logging.StreamHandler()
	console_handler.setFormatter(formatter)
	console_handler.setLevel(logging.INFO)
	logger.addHandler(console_handler)
	return logger

def close_logger(logger):
	handlers = logger.handlers[:]
	for handler in handlers:
		handler.close()
		logger.removeHandler(handler)
	logging.shutdown()
 
def cal_nei_index(ei, k, num_nodes, include_self=1):
    if include_self:
        neigh_dict = [k_hop_subgraph(node_idx=id, num_hops=k, edge_index=ei, num_nodes=num_nodes)[0] for id in range(num_nodes)]
    else:
        neigh_dict = [k_hop_subgraph(node_idx=id, num_hops=k, edge_index=ei, num_nodes=num_nodes)[0][1:] for id in range(num_nodes)]
    return neigh_dict

def cal_nc(nei_dict, y, thres=2.):
    device = y.device
    n = y.shape[0]
    
    # 收集所有非空邻居及其对应的i索引
    all_neigh_list = []
    i_indices_list = []
    for i in range(n):
        neigh = nei_dict[i]
        if len(neigh) > 0:
            all_neigh_list.append(neigh)
            i_indices_list.append(torch.full((len(neigh),), i, dtype=torch.long, device=device))
        
    if not all_neigh_list:  # 所有邻居都为空
        nc = torch.ones(n, device=device)
    else:
        all_neigh = torch.cat(all_neigh_list)
        i_indices = torch.cat(i_indices_list)
            
        # 获取所有邻居的标签并计算类别统计
        all_labels = y[all_neigh]
        sum_per_class = torch_scatter.scatter_add(all_labels, i_indices, dim=0, dim_size=n)
        max_counts = sum_per_class.max(dim=1).values
            
        # 计算每个节点的邻居数量
        k = torch.tensor([len(nei_dict[i]) for i in range(n)], dtype=torch.float, device=device)
            
        # 计算NC值并处理空邻居情况
        nc = torch.where(k > 0, k / max_counts, torch.tensor(1.0, device=device))
        
    mask = (nc <= thres).float().to(device)
    
    return mask

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

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


