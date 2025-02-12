from sympy import det
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

# NOTE: index is test_idx (int), not test_mask (bool)

def accuracy(output, labels):
	preds = output.argmax(dim=1)
	correct = (preds == labels).sum().item()
	return correct / labels.size(0)

def micro_f1(output, labels, index):
	preds = output[index].argmax(dim=1)
	return f1_score(labels[index].cpu().numpy(), preds.cpu().numpy(), average='micro')

def macro_f1(output, labels, index):
	preds = output[index].argmax(dim=1)
	return f1_score(labels[index].cpu().numpy(), preds.cpu().numpy(), average='macro')

def bacc(output, labels, index):
	preds = output.argmax(dim=1)
	return balanced_accuracy_score(labels[index].cpu().numpy(), preds[index].cpu().numpy())

def roc_auc(output, labels, index):
	one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=-1).to(index.device)
	detach_out = output[index].detach()
	num_classes = one_hot_labels.size(1)
	if num_classes > 2:
		return roc_auc_score(one_hot_labels[index].cpu().numpy(), detach_out.cpu().numpy(), multi_class="ovo")
	else:
		return roc_auc_score(one_hot_labels[index].cpu().numpy(), detach_out.cpu().numpy())

def per_class_acc(output, labels, index):
	preds = output[index].argmax(dim=1)
	labels_selected = labels[index]
	unique_labels = torch.unique(labels_selected)
	accuracy_per_class = {}
	for label in unique_labels:
		label_mask = (labels_selected == label)
		correct_predictions = (preds[label_mask] == labels_selected[label_mask]).sum().item()
		total_samples = label_mask.sum().item()
		if total_samples > 0:
			accuracy_per_class[label.item()] = correct_predictions / total_samples
		else:
			accuracy_per_class[label.item()] = 0.0

	return accuracy_per_class

def headtail_acc(output, labels, index, tail_mask):
	# NOTE: if use data augmention, tail_mask use the original size
	# We don't care if an augmented graph is head/tail size
	test_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
	test_mask[index] = True
	if tail_mask.size(0) < test_mask.size(0):
		aug_size = test_mask.size(0) - tail_mask.size(0)
		padding = torch.full((aug_size,), False, dtype=torch.bool, device=tail_mask.device)
		new_tail_mask = torch.cat((tail_mask,padding))
		# last (aug_size) items of test_mask is all False, because they are training items
		assert (test_mask[-aug_size:] == False).all()
	else:
		new_tail_mask = tail_mask

	# head/tail-size graphs in test_mask
	head_test = test_mask & (~new_tail_mask)
	tail_test = test_mask & new_tail_mask

	num_head = torch.sum(head_test).cpu().item()
	num_tail = torch.sum(tail_test).cpu().item()

	head_preds = output[head_test].argmax(dim=1)
	head_label = labels[head_test]

	tail_preds = output[tail_test].argmax(dim=1)
	tail_label = labels[tail_test]
	
	head_correct = (head_preds == head_label).sum().item()
	tail_correct = (tail_preds == tail_label).sum().item()

	head_acc = head_correct / num_head
	tail_acc = tail_correct / num_tail

	return {
		"head_acc" : head_acc,
		"tail_acc" : tail_acc,
	}