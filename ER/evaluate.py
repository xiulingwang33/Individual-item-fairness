import numpy as np
import torch
import pickle
import os


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

def recall(gt_item, pred_items):
	hits=0
	if gt_item in pred_items:
		hits+=1
		return hits
	return 0

def precision(gt_item, pred_items):
	hits=0
	if gt_item in pred_items:
		hits+=1
		return hits/10
	return 0

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def metrics(model, test_loader, top_k, device):
	HR, NDCG, RECALL, PREC = [], [], [], []
	# gt_items=[]
	# recommendss=[]

	for user, item, label in test_loader:
		user = user.to(device=device)
		item = item.to(device=device)

		predictions = model(user, item)

		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		RECALL.append(recall(gt_item, recommends))
		PREC.append(precision(gt_item, recommends))

		# gt_items.append(gt_item)
		# recommendss.append(recommends)

	# pickle.dump(recommendss,open("./{}_rec.pkl".format(epoch), "wb"))
	# pickle.dump(gt_items, open("./{}_gt.pkl".format(epoch), "wb"))

	return np.mean(HR), np.mean(NDCG), np.mean(RECALL), np.mean(PREC)

def metrics1(epoch,method,model, test_loader, top_k, device):
	HR, NDCG,RECALL,PREC = [], [],[],[]
	gt_items=[]
	recommendss=[]
	print(test_loader)
	for user, item, label in test_loader:
		user = user.to(device=device)
		item = item.to(device=device)
		#print(user)
		#print(item)
		predictions = model(user, item)
		#print(predictions)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()
		#print(recommends)
		gt_item = item[0].item()
		#print(item[0],gt_item)
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		RECALL.append(recall(gt_item, recommends))
		PREC.append(precision(gt_item, recommends))

		gt_items.append(gt_item)
		recommendss.append(recommends)
	dir_p="./result_{}/{}_rec_{}.pkl".format(method,epoch,method)
	ensureDir(dir_p)
	pickle.dump(recommendss,open("./result_{}/{}_rec_{}.pkl".format(method,epoch,method), "wb"))
	#pickle.dump(gt_items, open("./{}_gt_{}.pkl".format(epoch,method), "wb"))

	return np.mean(HR), np.mean(NDCG),np.mean(RECALL), np.mean(PREC)



def metrics2(epoch,method,model, test_loader, top_k, device):
	HR, NDCG,RECALL,PREC = [], [],[],[]
	gt_items=[]
	recommendss=[]
	print(test_loader)
	for user, item, label in test_loader:
		user = user.to(device=device)
		item = item.to(device=device)
		#print(user)
		#print(item)
		predictions = model(user, item)
		#print(predictions)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(item, indices).cpu().numpy().tolist()
		#print(recommends)
		gt_item = item[0].item()
		#print(item[0],gt_item)
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))
		RECALL.append(recall(gt_item, recommends))
		PREC.append(precision(gt_item, recommends))

		gt_items.append(gt_item)
		recommendss.append(recommends)
	dir_p="./result_{}/{}_rec_{}_.pkl".format(method,epoch,method)
	ensureDir(dir_p)
	pickle.dump(recommendss,open("./result_{}/{}_rec_{}_.pkl".format(method,epoch,method), "wb"))
	#pickle.dump(gt_items, open("./{}_gt_{}.pkl".format(epoch,method), "wb"))

	return np.mean(HR), np.mean(NDCG),np.mean(RECALL), np.mean(PREC)
