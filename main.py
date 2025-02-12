import warnings
import os
import torch
from datasets import DataLoader
from utils import gpr_splits
from models import *
import argparse
from parse import get_ncgnn_args
import numpy as np
import time
from config import seed_everything

from utils import (start_logger, close_logger, cal_nei_index, cal_nc)
from metrics import accuracy, micro_f1, macro_f1, roc_auc, bacc, per_class_acc, headtail_acc

def train(data, models):
    model = models["model"]
    criterion = models["criterion"]
    optimizer = models["optimizer"]
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


@torch.no_grad()
def test(data, models):
    model = models["model"]
    criterion = models["criterion"]
    # optimizer = models["optimizer"]
    model.eval()
    out = model(data)
    test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
    test_acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    test_bacc = bacc(out, data.y, data.test_mask)
    test_maf1 = macro_f1(out, data.y, data.test_mask)
    test_mif1 = micro_f1(out, data.y, data.test_mask)
    # test_rocauc = roc_auc(out, data.y, data.test_mask)
    acc_per_class = per_class_acc(out, data.y, data.test_mask)
    #acc_headtail = headtail_acc(out, data.y, data.test_mask, data.tail_mask)
    test_results = {
        "loss": test_loss,
        "acc": test_acc,
        "bacc": test_bacc,
        "micro_f1": test_mif1,
        "macro_f1": test_maf1,
        # "rocauc": test_rocauc, #TODO
        "rocauc": 0.00,
        "acc_per_class": acc_per_class,
        #"acc_headtail": acc_headtail,
    }
    return test_results

def test_ncgnn_model(logger, data, models):
    test_results = test(data, models)
    log =   "Test set results: " + \
            "loss: {:.4f} ".format(test_results["loss"].item()) + \
            "accuracy: {:.4f} ".format(test_results["acc"]) + \
            "micro_f1: {:.4f} ".format(test_results["micro_f1"]) + \
            "macro_f1: {:.4f} ".format(test_results["macro_f1"]) + \
            "rocauc: {:.4f} ".format(test_results["rocauc"]) + \
            "bacc: {:.4f} ".format(test_results["bacc"])
    logger.info(log)
    for cls in test_results["acc_per_class"].keys():
        acc = test_results["acc_per_class"][cls]
        logger.info(f"test_acc of class #{cls} : {acc:4f}")
    return test_results

@torch.no_grad()
def run_full_data(data, models, forcing=0):
    model = models["model"]
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
def valid(data, models):
    model = models["model"]
    criterion = models["criterion"]
    model.eval()
    out = model(data)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    val_acc = accuracy(out[data.val_mask], data.y[data.val_mask])
    val_bacc = bacc(out, data.y, data.val_mask)
    val_maf1 = macro_f1(out, data.y, data.val_mask)
    return val_loss, val_acc, val_bacc, val_maf1

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
    
    # TODO: 还要 data.tail_mask
    return data

def train_ncgnn_model(args, logger, data, models, best_acc, best_loss):
    best_flag = False

    t1 = time.time()
    loss, out = train(data, models)
    train_acc = accuracy(out[data.train_mask], data.y[data.train_mask])
    data.update_cc = False
    val_loss, val_acc, val_bacc, val_maf1 = valid(data, models)

    if val_acc > best_acc:
        logger.info('New best valid accuracy of backbone')
        best_acc, best_loss = val_acc, val_loss
        best_flag = True

        predict = run_full_data(data, models, args.forcing)
        neigh_dict = cal_nei_index(data.edge_index, args.hops, num_nodes)
        data.cc_mask = cal_nc(neigh_dict, predict.detach(), args.threshold) # 缓存上一次的
        data.update_cc = True
    else:
        best_flag = False

    t2 = time.time()
    epoch_time = t2 - t1
    log = f'Backbone epoch: {epoch + 1:d} loss_train: {loss.item():.4f} loss_val: {val_loss:.4f} train_acc: {train_acc:.4f} valid_acc: {val_acc:.4f} valid_macrof1: {val_maf1:.4f} valid_bacc: {val_bacc:.4f} time: {epoch_time:.3f}'
    logger.info(log)
    return best_acc, best_loss, best_flag

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = get_ncgnn_args()
    num_nodes, num_classes, num_features, data = DataLoader(args.dataset)
    # print(data) # x,y,edge_index
    # print(data.x.dtype) # torch.float32
    # print(data.y.dtype) # torch.int64
    # print(data.edge_index.dtype) # torch.int64
    # print(f"load {args.dataset} successfully!")
    # print('==============================================================')
    
    args_dict = vars(args)
    args = argparse.Namespace(**args_dict)
    args.threshold = 2 ** (args.threshold / 10 * np.log2(num_classes))
    
    logger = start_logger()

    logger.info(args)

    device = torch.device(f"cuda:{args.device}")

    data = load_ncgnn_datas(args, device, num_nodes, num_classes, data)

    models = load_ncgnn_models(num_features, num_classes, args, device)

    best_bak_valacc = 0.0
    best_bak_loss = 1e5

    for epoch in range(args.epochs):
        best_bak_valacc, best_bak_loss, best_flag = train_ncgnn_model(args, logger, data, models, best_bak_valacc, best_bak_loss)
        test_results = test_ncgnn_model(logger, data, models)
        # TODO: other backbone models
        if 'best_test_results' not in locals() or best_flag is True:
            best_test_results = test_results
        # # TODO: Early stop Method
        # if best_bak_valacc >= args.early_stop_acc:
        #     break

    if 'best_test_results' in locals():
        final_log = "Final Test set results (when valid_acc of backbone is best):\n" + \
				"loss: {:.4f} ".format(best_test_results["loss"].item()) + \
				"accuracy: {:.4f} ".format(best_test_results["acc"]) + \
				"micro_f1: {:.4f} ".format(best_test_results["micro_f1"]) + \
				"macro_f1: {:.4f} ".format(best_test_results["macro_f1"]) + \
				"rocauc: {:.4f} ".format(best_test_results["rocauc"]) + \
				"bacc: {:.4f} ".format(best_test_results["bacc"])
        logger.info(final_log)

    close_logger(logger)