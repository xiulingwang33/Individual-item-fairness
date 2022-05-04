from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import threading

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from utils import *

from numpy.linalg import norm
import find_seg



def segment_exist(path):
    return os.path.isfile(path)
def update_para(i, cov_list,cov_avg,beta):
    cov_max = max(cov_list)
    cov_min = min(cov_list)
    cov_high = min(cov_max, int(cov_avg + beta))
    cov_low = int(cov_high - beta)
    C_0 = []  # cov_max > coverage > cov_min
    C_00 = []  # coverage==cov_low, can't be substituted
    C_1 = []  # coverage>cov_high
    C_2 = []  # coverage<cov_min
    for item in i:
        if (cov_list[item] < cov_low):
            C_2.append(item)
        elif (cov_list[item] > cov_high):
            C_1.append(item)
        elif (cov_list[item] == cov_low):
            C_00.append(item)
        else:
            C_0.append(item)

    return (C_0, C_00, C_1, C_2, cov_high, cov_low, cov_max, cov_min)

def threshold(size, const):
    return (const * 1.0 / size)
def build_graph(product_embeds):
    similarity = np.dot(product_embeds, product_embeds.T)
    norm_sim = similarity
    for i in range(np.shape(similarity)[0]):
        for j in range (i,np.shape(similarity)[1]):
            ebd1=product_embeds[i]
            ebd2 = product_embeds[j]
            norm1 = norm(ebd1)
            norm2 = norm(ebd2)
            end_norm = float(norm1*norm2)
            norm_sim[i][j]=float(similarity[i][j]/end_norm )
            norm_sim[j][i]=norm_sim[i][j]
            #print(norm_sim[i][j])
    print(norm_sim)
    print(np.shape(norm_sim))

    num_nodes=np.shape(norm_sim)[0]
    similarity_list = []
    graph_edges = []
    print(np.shape(norm_sim)[0])
    print(np.shape(norm_sim)[1])
    for i in range(np.shape(norm_sim)[0]):
        for j in range(np.shape(norm_sim)[1]):
            if (j>i):
                similarity_ = [i, j, norm_sim[i][j]]
                similarity_list.append(similarity_)
                graph_edges.append(similarity_)
                #similarity_list = sorted(similarity_list, key=lambda x: x[2], reverse=True)  # from large to smallest
    return graph_edges,num_nodes,norm_sim
def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100
    avg_ndcg = np.mean(ndcgs) * 100
    avg_hit = np.mean(hits) * 100
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    for hop in range(3):
        state_tensor = torch.FloatTensor(state_pool).to(device)
        acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
        actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
        actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
        probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
        probs = probs + actmask_tensor.float()  # In order to differ from masked actions
        topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
        topk_idxs = topk_idxs.detach().cpu().numpy()
        topk_probs = topk_probs.detach().cpu().numpy()

        new_path_pool, new_probs_pool = [], []
        for row in range(topk_idxs.shape[0]):
            path = path_pool[row]
            probs = probs_pool[row]
            for idx, p in zip(topk_idxs[row], topk_probs[row]):
                if idx >= len(acts_pool[row]):  # act idx is invalid
                    continue
                relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                if relation == SELF_LOOP:
                    next_node_type = path[-1][1]
                else:
                    next_node_type = KG_RELATION[path[-1][1]][relation]
                new_path = path + [(relation, next_node_type, next_node_id)]
                new_path_pool.append(new_path)
                new_probs_pool.append(probs + [p])
        path_pool = new_path_pool
        probs_pool = new_probs_pool
        if hop < 2:
            state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    print('Predicting paths...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))


def evaluate_paths(path_file, train_labels, test_labels,K,min_size,threshold_func):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[PURCHASE][0]
    product_embeds = embeds[PRODUCT]
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)
    alpha=5e-05
    comp_sets_noseg_file = '{}/comp_sets_noseg_{}.txt'.format(TMP_DIR[args.dataset],alpha)
    beta_rate = 0.01
    #

    sorted_graph = pickle.load(open('{}/sorted_graph.pkl'.format(TMP_DIR[args.dataset]), 'rb'))
    ####the items pairs sorted by embedding similarity value


    if not segment_exist(comp_sets_noseg_file):


        sim_edge_num=int(alpha*np.shape(sorted_graph)[0])
        sim_alpha=sorted_graph[sim_edge_num][2]
        print(sim_alpha)
        select_edge=sorted_graph[0:sim_edge_num]
        comp_sets_noseg=find_seg.find_seg(select_edge)
        


        comp_sets_noseg_file = '{}/comp_sets_noseg_{}.pkl'.format(TMP_DIR[args.dataset],alpha)
        pickle.dump(comp_sets_noseg, open(comp_sets_noseg_file, 'wb'))

        out = open('{}/comp_sets_noseg_{}.txt'.format(TMP_DIR[args.dataset],alpha), 'w')
        for comp in comp_sets_noseg:
            for jtem in comp:
                out.write(str(jtem) + '\t')
            out.write('\n')
        out.close()

    else:
        comp_sets_noseg_file = '{}/comp_sets_noseg_{}.pkl'.format(TMP_DIR[args.dataset],alpha)
        comp_sets_noseg=pickle.load(open(comp_sets_noseg_file, 'rb'))



    pred_paths = pickle.load(open('{}/pred_paths_post.pkl'.format(TMP_DIR[args.dataset]), 'rb'))
    sorted_path = pickle.load(open('{}/sorted_path_post.pkl'.format(TMP_DIR[args.dataset]), 'rb'))
    best_pred_paths = pickle.load(open('{}/best_pred_paths_post.pkl'.format(TMP_DIR[args.dataset]), 'rb'))
    if args.dataset=='beauty':
        pred_labels = pickle.load(open('./item-fairness-combined/0_rec_100227-pgpr3{}-{}-128-3-5.pkl'.format(alpha, beta_rate), 'rb'))
    if args.dataset=='cell':
        pred_labels = pickle.load(open('./item-fairness-combined/0_rec_101227-pgpr3{}-{}-256-3-5.pkl'.format(alpha, beta_rate), 'rb'))
    test_uids = list(test_labels.keys())
    test_uids = sorted(test_uids)
    num_test_uids = np.shape(test_uids)[0]
    num_test_pids = np.shape(product_embeds)[0]
    cov_matrix = np.zeros((num_test_uids, num_test_pids), int)
    for uid in best_pred_paths:
        for pid in pred_labels[uid]:
            cov_matrix[uid][pid] += 1
    cnt = 0
    subs=[]
    unfair_cnt = 0
    unfair_pairs=[]
    fair_cnt = 0
    fair_pairs = []
    exist_path_cover_lists=[]
    exist_path_cover_list = np.zeros((np.shape(product_embeds)[0]),
                                     int)  # coverage of each item if there is a reachable path, except already buy
    for comp1 in comp_sets_noseg:


        if args.dataset=='cell':
            out = '{}/exist_path_cover_list.txt'.format(TMP_DIR[args.dataset])
            lines = open(out, 'r').readlines()
            for l in lines:
         
                tmps = l.strip()
                inters = [int(i) for i in tmps.split('\t')]
                exist_path_cover_lists.append(inters)
        if args.dataset=='beauty':
            exist_path_cover_list_file = '{}/exist_path_cover_list.pkl'.format(TMP_DIR[args.dataset])
            exist_path_cover_list = pickle.load(open(exist_path_cover_list_file, 'rb'))

        cov_sum = 0
        item_cnt=0
        exist_path_cover_sum = 0
        Ms=[]
        print(comp1)
        cov_list = np.zeros((np.shape(product_embeds)[0]), int)  # coverage of each item
        comp=comp1

        for item in comp:
            item_cnt+=1
            print('$', item)
            cov_sum += cov_matrix[:, item].sum()
            cov_list[item] = int(cov_matrix[:, item].sum())


        for item in comp:
            item_cnt+=1
            cov_sum += cov_matrix[:, item].sum()
            cov_list[item] = int(cov_matrix[:, item].sum())



            exist_path_cover_sum += exist_path_cover_list[item]

        beta = float(2*beta_rate * exist_path_cover_sum/item_cnt)

        com_edge=[]
        for i in range(int(alpha*np.shape(sorted_graph)[0])):

            edge1=sorted_graph[i]
            com_edge.append([edge1[0],edge1[1]])
        
        for i in comp:
            for j in comp:
                if ([i,j]) in com_edge:
                    M=exist_path_cover_list[i]+exist_path_cover_list[j]
 
                    if (float(abs(cov_list[i]-cov_list[j])/M)>beta_rate):
                        unfair_cnt+=1
                        unfair_pair=[i,j,cov_list[i],cov_list[j],exist_path_cover_list[i],exist_path_cover_list[j]]
                        unfair_pairs.append(unfair_pair)
                    else:
                        fair_cnt += 1
                        fair_pair = [i, j, cov_list[i], cov_list[j], exist_path_cover_list[i],exist_path_cover_list[j]]
                        fair_pairs.append(fair_pair)
                    Ms.append(M)


        cov_avg = cov_sum / (len(comp))
        C_0, C_00, C_1, C_2, cov_high, cov_low, cov_max, cov_min = update_para(comp, cov_list, cov_avg, beta)
        print(C_0, C_00, C_1, C_2)
        C_r = []
        C_q = []


        while ((C_1 != [] and C_2 != []) or (C_1 != [] and C_0 != []) or (C_2 != [] and C_0 != [])):
            print(C_1, C_2, C_0, C_00, cov_high, cov_low, cov_max, cov_min, beta_rate)
            cost_min = 1000000
            if (C_1 != [] and C_2 != []):
                C_r = C_2
                C_q = C_1
                # flag=0
            elif (C_1 != [] and C_0 != []):
                C_r = C_1
                C_q = C_0
                # flag=1
            elif (C_2 != [] and C_0 != []):
                C_r = C_2
                C_q = C_0
                # flag=2

            for i in C_r:
                for j in C_q:
                    print(cov_list[i], cov_list[j])

                    for uid in best_pred_paths:
                        for pid in pred_labels[uid]:



                            if (pid == j):
                                cost = abs(scores[uid][j] - scores[uid][i])  #
                                print(cost)
                                if (cost < cost_min):
                                    sub_uid = uid
                                    sub_pid = i
                                    under_sub_pid = j
                                    cost_min = cost
            # update C_ after substitution

            sub=[sub_uid, sub_pid, under_sub_pid, cost_min]
            subs.append(sub)

            cnt += 1


            l = pred_labels[sub_uid].index(under_sub_pid)

            pred_labels[sub_uid][l] = sub_pid


            cov_matrix[sub_uid][sub_pid] += 1
            cov_matrix[sub_uid][under_sub_pid] -= 1

            cov_list[sub_pid] += 1
            cov_list[under_sub_pid] -= 1

            C_0, C_00, C_1, C_2, cov_high, cov_low, cov_max, cov_min = update_para(comp, cov_list, cov_avg, beta)

    pred_labels_file = '{}/pred_labels_post_noseg_{}_{}_10.16.pkl'.format(TMP_DIR[args.dataset], beta_rate,alpha)
    pickle.dump(pred_labels, open(pred_labels_file, 'wb'))
    evaluate(pred_labels, test_labels)

    

    out = open('{}/subs_noseg_{}_{}.txt'.format(TMP_DIR[args.dataset],beta_rate,alpha), 'w')
    for isub in subs:
        out.write(str(isub))
        out.write('\n')
    out.close()



    out = open('{}/exist_path_cover_lists_noseg_{}_{}.txt'.format(TMP_DIR[args.dataset], beta_rate,alpha), 'w')
    for item in exist_path_cover_lists:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('{}/unfair_pairs_{}_{}.txt'.format(TMP_DIR[args.dataset], beta_rate,alpha), 'w')
    for item in unfair_pairs:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()

    out = open('{}/fair_pairs_{}_{}.txt'.format(TMP_DIR[args.dataset], beta_rate,alpha), 'w')
    for item in fair_pairs:
        for jtem in item:
            out.write(str(jtem) + '\t')
        out.write('\n')
    out.close()





def test(args):
    policy_file = args.log_dir + '/policy_model_epoch_{}.ckpt'.format(args.epochs)
    path_file = args.log_dir + '/policy_paths_epoch{}.pkl'.format(args.epochs)

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    #if args.run_path:
        #predict_paths(policy_file, path_file, args)
    if args.run_eval:
        evaluate_paths(path_file, train_labels, test_labels,args.K,args.min_comp_size,threshold)


if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='number of samples')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    parser.add_argument('--K', type=float, default=1.0,
                        help='a constant to control the threshold function of the predicate')
    parser.add_argument('--min-comp-size', type=int, default=1,
                        help='a constant to remove all the components with fewer number of pixels')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    test(args)


