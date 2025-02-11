import warnings
import os
import torch
import torch_scatter
from tqdm import trange
from torch_geometric.utils import k_hop_subgraph
from datasets import DataLoader
from utils import gpr_splits
from models import *
import argparse
from parse import get_ncgnn_args
import numpy as np
import time
from config import seed_everything


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


@torch.no_grad()
def test(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    return test_acc


@torch.no_grad()
def run_full_data(data, forcing=0):
    model.eval()
    mask = data.train_mask.detach().view(-1, 1)
    out = model(data)
    pred = out.argmax(dim=1, keepdim=True)  # Use the class with highest probability.
    if forcing:
        print('use forcing...')
        pred = data.y.detach().view(-1, 1) * mask + pred * ~mask
    onehot = torch.zeros(out.shape, device=device)
    onehot.scatter_(1, pred, 1)

    return onehot


@torch.no_grad()
def valid(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc


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

def load_ncgnn_models(num_features, num_classes, args, device):
    model = NCGCN(num_features, num_classes, args).to(device)
    model.reset_parameters()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    models = {
        "model" : model,
        "criterion" : criterion,
        "optimizer" : optimizer,
    }
    
    return models

def load_ncgnn_datas(args, device, num_nodes, num_classes, data):
    t1 = time.time()
    train_rate = 0.6
    val_rate = 0.2
    # dataset split
    if args.dataset == 'penn94':
        num_nodes = torch.count_nonzero(data.y + 1).item()

    percls_trn = int(round(train_rate * num_nodes / num_classes))
    val_lb = int(round(val_rate * num_nodes))
    data.x = SparseTensor.from_dense(data.x)
    # 10 times rand part
    seed_everything(args.seed)
    #neigh_dict = cal_nei_index(data.edge_index, args.hops, num_nodes)
    #print('indexing finished')
  
    # training settings
    # seed_everything(args.seed)
    data.cc_mask = torch.ones(num_nodes).float()
    data = gpr_splits(data, num_nodes, num_classes, percls_trn, val_lb).to(device) # 3 mask
    data.update_cc = True
    
    t2 = time.time()
    print("load datas time:",t2-t1)
    return data

if __name__ == "__main__":
    args = get_ncgnn_args()

    num_nodes, num_classes, num_features, data = DataLoader(args.dataset)

    # print(data) # x,y,edge_index
    # print(data.x.dtype) # torch.float32
    # print(data.y.dtype) # torch.int64
    # print(data.edge_index.dtype) # torch.int64
    # print(f"load {args.dataset} successfully!")
    # print('==============================================================')
    # warnings.filterwarnings("ignore")
    args_dict = vars(args)
    args = argparse.Namespace(**args_dict)
    args.threshold = 2 ** (args.threshold / 10 * np.log2(num_classes))
    print(args)

    device = torch.device(f"cuda:{args.device}")

    data = load_ncgnn_datas(args, device, num_nodes, num_classes, data)

    models = load_ncgnn_models(num_features, num_classes, args, device)
    model = models["model"]
    criterion = models["criterion"]
    optimizer = models["optimizer"]

    accs, test_accs = [], []
    ep_list = []
    best_acc = 0.
    final_test_acc = 0.
    # es_count = patience = 100
    for epoch in range(500):
        t1 = time.time()
        loss, out = train(data)
        data.update_cc = False
        val_acc = valid(data)
        test_acc = test(data)
        if val_acc > best_acc:
            # es_count = patience
            best_acc = val_acc
            final_test_acc = test_acc
            predict = run_full_data(data, args.forcing)
            neigh_dict = cal_nei_index(data.edge_index, args.hops, num_nodes)
            data.cc_mask = cal_nc(neigh_dict, predict.detach(), args.threshold) # 缓存上一次的
            data.update_cc = True
        t2 = time.time()
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Time: {t2 - t1:.4f}s")
        # else:
        #     es_count -= 1
        # if es_count <= 0:
        #     break
    print("Test acc:",final_test_acc)