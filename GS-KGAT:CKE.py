from __future__ import absolute_import, division, print_function

import os

from math import log

from numpy.linalg import norm

import pickle
import numpy as np


def segment_exist(path):
    return os.path.isfile(path)


def update_para(i, cov_list, cov_avg, beta):
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
        for j in range(i, np.shape(similarity)[1]):
            ebd1 = product_embeds[i]
            ebd2 = product_embeds[j]
            norm1 = norm(ebd1)
            norm2 = norm(ebd2)
            end_norm = float(norm1 * norm2)
            norm_sim[i][j] = float(similarity[i][j] / end_norm)
            norm_sim[j][i] = norm_sim[i][j]


    num_nodes = np.shape(norm_sim)[0]
    similarity_list = []
    graph_edges = []
    print(np.shape(norm_sim)[0])
    print(np.shape(norm_sim)[1])
    for i in range(np.shape(norm_sim)[0]):
        for j in range(np.shape(norm_sim)[1]):
            if (j > i):
                similarity_ = [i, j, norm_sim[i][j]]
                similarity_list.append(similarity_)
                graph_edges.append(similarity_)
                # similarity_list = sorted(similarity_list, key=lambda x: x[2], reverse=True)  # from large to smallest
    return graph_edges, num_nodes, norm_sim


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


def _load_ratings(file_name):
    user_dict = dict()
    inter_mat = list()

    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))

        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])

        if len(pos_ids) > 0:
            user_dict[u_id] = pos_ids
    return np.array(inter_mat), user_dict


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument('--alpha', type=float, default=1e-05,
                        help='alpha.')
    parser.add_argument('--beta_rate', type=float, default=0.1,
                        help='KG Embedding size.')
    parser.add_argument('--dataset', nargs='?', default='amazon-beauty',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--model_type', nargs='?', default='cke',
                        help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')

    return parser.parse_args()


args = parse_args()
model_type = args.model_type
dataset = args.dataset
alpha = args.alpha
comp_sets_noseg_file = '../Data/{}/comp_sets_noseg_{}.pkl'.format(dataset, alpha)
comp_sets_noseg = pickle.load(open(comp_sets_noseg_file, 'rb'))

test_labels_file = '../Data/{}/test_label.pkl'.format(dataset)
test_labels = pickle.load(open(test_labels_file, 'rb'))

###the recommendation list from KGAT/CKE

best_pred_items = pickle.load(open('./{}_{}_best_pred_items.pkl'.format(dataset, model_type), 'rb'))
user_batch_rating_uid_maps1 = pickle.load(
    open('./{}_{}_user_batch_rating_uid_maps.pkl'.format(dataset, model_type), 'rb'))

train_file = '../Data/{}/train.txt'.format(dataset)
test_file = '../Data/{}/test.txt'.format(dataset)
train_data, train_user_dict = _load_ratings(train_file)
# print(train_user_dict)
test_data, test_user_dict = _load_ratings(test_file)
# print(test_user_dict)
user_batch_rating_uid_maps_file = './{}_{}_user_batch_rating_uid_maps.pkl'.format(dataset, model_type)
if not segment_exist(user_batch_rating_uid_maps_file):
    user_batch_rating_uid_maps = []
    for user_batch_rating in user_batch_rating_uid_maps1:
        u = user_batch_rating[0]
        items_score = user_batch_rating[1]
        for item in train_user_dict[u]:
            items_score[item] = -9999999999999999
        user_batch_rating_uid_maps.append([u, items_score])

    user_batch_rating_uid_maps_file = './{}_{}_user_batch_rating_uid_maps.pkl'.format(dataset, model_type)
    pickle.dump(user_batch_rating_uid_maps, open(user_batch_rating_uid_maps_file, 'wb'))

else:
    user_batch_rating_uid_maps = pickle.load(open(user_batch_rating_uid_maps_file, 'rb'))

num_test_uids = np.shape(user_batch_rating_uid_maps)[0]
num_test_pids = len(user_batch_rating_uid_maps[0][1])
cov_matrix = np.zeros((num_test_uids, num_test_pids), int)

best_pred_items1 = []
if args.dataset == 'amazon-beauty':
    pred_labels = pickle.load(
        open('./item-fairness-combined/0_rec_100227-{}3{}-{}-128-3-5.pkl'.format(model_type,alpha, args.beta_rate), 'rb'))
if args.dataset == 'amazon-cellphone':
    pred_labels = pickle.load(
        open('./item-fairness-combined/0_rec_101227-{}3{}-{}-256-3-5.pkl'.format(model_type,alpha, args.beta_rate), 'rb'))

for i in range(np.shape(pred_labels)[0]):
    for pid in pred_labels[i]:
        cov_matrix[i][pid] += 1
    best_pred_items1.append([i, pred_labels[i]])


cnt = 0
subs = []
unfair_cnt = 0
unfair_pairs = []
fair_cnt = 0
fair_pairs = []

exist_path_cover_lists = []
out = '../Data/{}/exist_path_cover_lists_noseg.txt'.format(dataset)
lines = open(out, 'r').readlines()
# print(lines)
for l in lines:
    # print(l)
    tmps = l.strip()
    # print(tmps)
    inters = [int(i) for i in tmps.split('\t')]
    exist_path_cover_lists.append(inters)
sorted_exist_path_cover_list = exist_path_cover_lists

pred_labels = {}
for comp1 in comp_sets_noseg:

    beta_rate = args.beta_rate

    cov_sum = 0
    item_cnt = 0
    exist_path_cover_sum = 0
    Ms = []
    print(comp1)
    cov_list = np.zeros(num_test_pids, int)  # coverage of each item
    comp = comp1

    for item in comp:
        item_cnt += 1

        cov_sum += cov_matrix[:, item].sum()
        cov_list[item] = int(cov_matrix[:, item].sum())

        exist_path_cover_sum += int(sorted_exist_path_cover_list[item][0])

    beta = float(2 * beta_rate * exist_path_cover_sum / item_cnt)

    com_edge = []

    sorted_graph = pickle.load(open('./{}_sorted_graph.pkl'.format(dataset), 'rb'))
    for i in range(int(alpha * np.shape(sorted_graph)[0])):
        edge1 = sorted_graph[i]
        com_edge.append([edge1[0], edge1[1]])

    for i in comp:
        for j in comp:
            if ([i, j]) in com_edge:
                for u in range(np.shape(cov_matrix)[0]):
                    M = 0
                    if (cov_matrix[u][i] + cov_matrix[u][j] == 1):
                        M = M + 2
                    elif (cov_matrix[u][i] + cov_matrix[u][j] == 2):
                        M = M + 1

                if (float(abs(cov_list[i] - cov_list[j]) / M) > beta_rate):
                    unfair_cnt += 1
                    unfair_pair = [i, j, cov_list[i], cov_list[j], exist_path_cover_lists[i], exist_path_cover_lists[j]]
                    unfair_pairs.append(unfair_pair)
                else:
                    fair_cnt += 1
                    fair_pair = [i, j, cov_list[i], cov_list[j], exist_path_cover_lists[i], exist_path_cover_lists[j]]
                    fair_pairs.append(fair_pair)
                Ms.append(M)

    cov_avg = cov_sum / (len(comp))
    C_0, C_00, C_1, C_2, cov_high, cov_low, cov_max, cov_min = update_para(comp, cov_list, cov_avg, beta)
    print(C_0, C_00, C_1, C_2)
    C_r = []
    C_q = []

    while ((C_1 != [] and C_2 != []) or (C_1 != [] and C_0 != []) or (C_2 != [] and C_0 != [])):
        #print(C_1, C_2, C_0, C_00, cov_high, cov_low, cov_max, cov_min, beta)
        cost_min = 9999999999999999
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

                for term in best_pred_items1:
                    uid = term[0]
                    # print(term[1])
                    for pid in term[1]:

                        if (pid == j):
                            cost = abs(user_batch_rating_uid_maps[uid][1][j] - user_batch_rating_uid_maps[uid][1][i])  #
                            #print(cost)
                            if (cost < cost_min):
                                sub_uid = uid
                                sub_pid = i
                                under_sub_pid = j
                                cost_min = cost
        sub = [sub_uid, sub_pid, under_sub_pid, cost_min]
        subs.append(sub)

        if np.shape(subs)[0] % 1000 == 0:
            out = open('./{}_{}_subs_noseg_{}_{}.txt'.format(dataset, model_type, beta_rate, alpha), 'w')
            for isub in subs:
                out.write(str(isub))
                out.write('\n')
            out.close()


        cnt += 1

        if under_sub_pid not in best_pred_items1[sub_uid][1]:
            C_1 = []
            C_2 = []
            C_0 = []
            continue;
        l = best_pred_items1[sub_uid][1].index(under_sub_pid)

        best_pred_items1[sub_uid][1][l] = sub_pid

        cov_matrix[sub_uid][sub_pid] += 1
        cov_matrix[sub_uid][under_sub_pid] -= 1

        cov_list[sub_pid] += 1
        cov_list[under_sub_pid] -= 1

        C_0, C_00, C_1, C_2, cov_high, cov_low, cov_max, cov_min = update_para(comp, cov_list, cov_avg, beta)

for x in best_pred_items1:
    pred_labels[x[0]] = (x[1])[0:10]
pred_labels_file = './{}_{}_pred_labels_post_noseg_{}_{}.pkl'.format(dataset, model_type,beta_rate, alpha)
pickle.dump(pred_labels, open(pred_labels_file, 'wb'))
evaluate(pred_labels, test_labels)



out = open('./{}_{}_subs_noseg_{}_{}.txt'.format(dataset,model_type, beta_rate, alpha), 'w')
for isub in subs:
    out.write(str(isub))
    out.write('\n')
out.close()

out = open('./{}_{}_exist_path_cover_lists_noseg_{}_{}.txt'.format(dataset, model_type,beta_rate, alpha), 'w')
for item in exist_path_cover_lists:
    for jtem in item:
        out.write(str(jtem) + '\t')
    out.write('\n')
out.close()









