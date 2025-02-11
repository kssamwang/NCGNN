import warnings
import os
from tqdm import trange
from torch_geometric.utils import k_hop_subgraph
from datasets import DataLoader
from utils import gpr_splits
from models import *
import argparse
from parse import get_ncgnn_args
import numpy as np

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
    if not os.path.exists('index'):
        os.makedirs('index')
    if include_self:
        path_name = f'index/{args.dataset}_hop{k}.npy'
    else:
        path_name = f'index/{args.dataset}_hop{k}_noself.npy'
    if os.path.exists(path_name):
        neigh_dict = np.load(path_name, allow_pickle=True).item()
    else:
        neigh_dict = {}
        for id in trange(num_nodes):
            # neigh = k_hop_subgraph(id, k, ei)[0]
            # exclude self
            if include_self:
                neigh = k_hop_subgraph(id, k, ei)[0]
            else:
                neigh = k_hop_subgraph(id, k, ei)[0][1:]
            neigh_dict[id] = neigh
        np.save(path_name, neigh_dict)
    return neigh_dict


def cal_nc(nei_dict, y, thres=2., use_tensor=True):
    if not use_tensor:
        y = y.numpy()
    nc = np.empty(y.shape[0])
    for i, neigh in nei_dict.items():
        if use_tensor:
            labels = torch.index_select(y, 0, neigh)
            if len(labels):
                nc[i] = len(labels) / torch.max(torch.sum(labels, dim=0)).item()
            else:
                nc[i] = 1.0
        else:
            labels = y[neigh].reshape(len(neigh))
            if len(labels):
                nc[i] = len(labels) / max(np.bincount(labels))
            else:
                nc[i] = 1.0

    # low_cc: 1 ; high_cc: 0
    mask = np.where(nc <= thres, 1., 0.)
    return torch.from_numpy(mask).float().to(device)

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
    neigh_dict = cal_nei_index(data.edge_index, args.hops, num_nodes)
    print('indexing finished')
  
    # training settings
    # seed_everything(args.seed)
    data.cc_mask = torch.ones(num_nodes).float()
    data = gpr_splits(data, num_nodes, num_classes, percls_trn, val_lb).to(device) # 生成三个mask
    data.update_cc = True
    
    datas = {
        "data" : data,
        "neigh_dict" : neigh_dict,
    }
    
    return datas

if __name__ == "__main__":
    args = get_ncgnn_args()

    num_nodes, num_classes, num_features, data = DataLoader(args.dataset)

    print(data) # x,y,edge_index
    print(data.x.dtype) # torch.float32
    print(data.y.dtype) # torch.int64
    print(data.edge_index.dtype) # torch.int64
    # print(f"load {args.dataset} successfully!")
    # print('==============================================================')
    # warnings.filterwarnings("ignore")
    args_dict = vars(args)
    args = argparse.Namespace(**args_dict)
    args.threshold = 2 ** (args.threshold / 10 * np.log2(num_classes))
    print(args)

    device = torch.device(f"cuda:{args.device}")

    datas = load_ncgnn_datas(args, device, num_nodes, num_classes, data)
    data = datas["data"]
    neigh_dict = datas["neigh_dict"]

    models = load_ncgnn_models(num_features, num_classes, args, device)
    model = models["model"]
    criterion = models["criterion"]
    optimizer = models["optimizer"]

    accs, test_accs = [], []
    ep_list = []
    best_acc = 0.
    final_test_acc = 0.
    es_count = patience = 100
    for epoch in range(500):
        loss, out = train(data)
        data.update_cc = False
        val_acc = valid(data)
        test_acc = test(data)
        if val_acc > best_acc:
            es_count = patience
            best_acc = val_acc
            final_test_acc = test_acc
            predict = run_full_data(data, args.forcing)
            data.cc_mask = cal_nc(neigh_dict, predict.detach().cpu(), args.threshold)
            data.update_cc = True
        else:
            es_count -= 1
        if es_count <= 0:
            break
    print("Test acc:",final_test_acc)