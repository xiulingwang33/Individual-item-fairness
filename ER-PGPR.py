import os
import time
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import data_util
import evaluate
import GMF
import MLP

import IndividualFairnessLoss

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epoches")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim_GMF", type=int, default=100, help="dimension of embedding in GMF submodel")
parser.add_argument("--embedding_dim_MLP", type=int, default=128, help="dimension of embedding in MLP submodel")
parser.add_argument("--hidden_layer_MLP", type=list, default=[128, 64, 32], help="hidden layers in MLP")
parser.add_argument("--use_pretrained", action="store_true", help="use pretrained model to initialize weights")
parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
parser.add_argument("--data_set", type=str, default="ml-1m", help="data set. 'ml-1m' or 'pinterest-20'")
parser.add_argument("--data_path", type=str, default="./data")
parser.add_argument("--model_path", type=str, default="./model")
parser.add_argument("--out", default=True, help="save model or not")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--alpha", type=float, default=1e-05)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--sample_size", type=int, default=50)
parser.add_argument('--lreward',dest='lambda_reward',type=float,default=1.0,help="Lambda for reward in the REINFORCE style updates.Can be set to 0 to start all fairness training")
parser.add_argument("--model_type", type=str, default="cke")
args = parser.parse_args()

args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.lambda_ind_fairness = 0.1
args.lambda_reward=1.0

cudnn.benchmark = True

def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(dataset)
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed
class NeuMF(nn.Module):
    def __init__(self, user_num, item_num, embedding_dim_GMF, embedding_dim_MLP, hidden_layer_MLP,
                 dropout, GMF_model=None, MLP_model=None):
        super(NeuMF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        embedding_dim_GMF: number of embedding dimensions in GMF submodel;
        embedding_dim_MLP: number of embedding dimensions in MLP submodel;
        hidden_layer_MLP: dimension of each hidden layer (list type) in MLP submodel;
        dropout: dropout rate between fully connected layers;
        GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        dataset='amazon_cellphone'
        embeds = load_embed(dataset)
        user_embeds = embeds['user'][0:user_num, :]
        purchase_embeds = embeds['purchase'][0]
        product_embeds = embeds['product'][0:item_num, :]
        user_embeds=user_embeds+purchase_embeds


        self.embed_user_GMF = nn.Embedding(user_num, embedding_dim_GMF)
        self.embed_item_GMF = nn.Embedding(item_num, embedding_dim_GMF)
        self.embed_user_MLP = nn.Embedding(user_num, embedding_dim_MLP)
        self.embed_item_MLP = nn.Embedding(item_num, embedding_dim_MLP)

        # for i in self.embed_user_GMF.parameters():
        #     i.requires_grad = False
        #
        # for i in self.embed_item_GMF.parameters():
        #     i.requires_grad = False

        MLP_modules = []
        self.num_layers = len(hidden_layer_MLP)
        for i in range(self.num_layers):
            MLP_modules.append(nn.Dropout(p=self.dropout))
            if i == 0:
                MLP_modules.append(nn.Linear(embedding_dim_MLP * 2, hidden_layer_MLP[0]))
            else:
                MLP_modules.append(nn.Linear(hidden_layer_MLP[i - 1], hidden_layer_MLP[i]))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(hidden_layer_MLP[-1] + embedding_dim_GMF, 1)

        # self.embed_user_GMF.weight=torch.from_numpy(user_embeds)
        # self.embed_item_GMF.weight=torch.from_numpy(product_embeds)

        self.embed_user_GMF.weight.data.copy_(torch.from_numpy(np.array(user_embeds)))
        self.embed_item_GMF.weight.data.copy_(torch.from_numpy(np.array(product_embeds)))


        if not args.use_pretrained:
            # nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            # nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            # Kaiming/Xavier initialization can not deal with non-zero bias terms
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item.weight)

            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)


    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF

        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


if __name__ == "__main__":
    model_type='pgpr'
    method = model_type+'-'+str(args.alpha)+'-' +str(args.beta)+'-'+str(args.batch_size)+'-'+str(args.lambda_ind_fairness)
    dataset = 'amazon_cellphone'

    test_lables = pickle.load(open('./{}/test_label.pkl'.format(dataset), 'rb'))
    test_user_products = test_lables

    train_lables = pickle.load(open('./{}/train_label.pkl'.format(dataset), 'rb'))
    train_user_products = train_lables

    num_test_uids = len(test_lables.keys())
    if (dataset == 'Amazon_Beauty'):
        num_test_pids = 12101
    if (dataset == 'Amazon_Cellphones'):
        num_test_pids = 10429

    # data_file = os.path.join(args.data_path, args.data_set)
    # train_data, test_data, user_num, item_num, train_mat = data_util.load_all(data_file)

    train_pid_list = pickle.load(open("./{}/train_pid_list.pkl".format(dataset), "rb"))
    train_fair_pid_list = pickle.load(open("./{}/train_fair_pid_list.pkl".format(dataset), "rb"))
    test_pid_list = pickle.load(open("./{}/test-101_pid_list_{}.pkl".format(dataset,model_type), "rb"))
    train_mat = np.zeros((num_test_uids, num_test_pids), int)

    # for uid in train_lables.keys():
    #     for pid in train_lables[uid]:
    #         train[uid][pid]=1
    train_data = []
    test_data = []
    # test_data_neg = []
    test_lists = {}
    test_list = []
    train_negs = {}
    train_loader = []
    test_loader = []
    train_fair_data=[]

    for uid in range(num_test_uids):
        train_neg = []
        train_neg_data = []
        for pid in range(num_test_pids):
            if pid in train_lables[uid]:
                train_mat[uid][pid] = 1
                train_data.append([uid, pid])
                # train_loader.append([uid,pid,1])

            elif pid in train_pid_list[uid]:
                train_neg.append(pid)
                train_neg_data.append([uid, pid])
                # train_loader.append([uid, pid, 0])


        train_negs[uid] = train_neg

        # test_lists[uid]=test_list

    for uid in range(num_test_uids):
        for pid in test_pid_list[uid]:
            if pid in train_lables[uid]:
                train_fair_data.append([uid, pid, 1])

            else:
                train_fair_data.append([uid, pid, 0])


    for uid in range(num_test_uids):
        for pid in test_pid_list[uid]:
            if pid in test_user_products[uid]:
                test_data.append([uid, pid, 1])

            else:
                test_data.append([uid, pid, 0])


    user_num = num_test_uids
    item_num = num_test_pids

    cnt_batch = int((np.shape(train_data)[0] + np.shape(train_neg_data)[0]) / args.batch_size) + 1

    train_dataset = data_util.NCFData1(train_data, train_negs, item_num, train_mat, args.num_ng, True)
    test_dataset = data_util.NCFData(test_data, item_num, train_mat, 0, False)
    train_fair_dataset = data_util.NCFData(train_fair_data, item_num, train_mat, 0, False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)
    train_fair_loader = data.DataLoader(train_fair_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    GMF_model_path = os.path.join(args.model_path, 'GMF.pth')
    MLP_model_path = os.path.join(args.model_path, 'MLP.pth')
    if args.use_pretrained:
        assert os.path.exists(GMF_model_path), 'lack of GMF model'
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = GMF.GMF(user_num, item_num, args.embedding_dim_GMF, args.dropout)
        GMF_model.load_state_dict(torch.load(GMF_model_path))

        MLP_model = MLP.MLP(user_num, item_num, args.embedding_dim_MLP, args.hidden_layer_MLP, args.dropout)
        MLP_model.load_state_dict(torch.load(MLP_model_path))
    else:
        GMF_model = None
        MLP_model = None
    model = NeuMF(user_num, item_num, args.embedding_dim_GMF, args.embedding_dim_MLP,
                  args.hidden_layer_MLP, args.dropout, GMF_model, MLP_model)
    model.to(device=args.device)
    loss_function = nn.BCEWithLogitsLoss()

    # loss.requires_grad = True
    if args.use_pretrained:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    sorted_graph = pickle.load(open('./{}/sorted_graph.pkl'.format(dataset), 'rb'))
    reachable_path_matrix = pickle.load(open('{}/reachable_path_matrix.pkl'.format(dataset), 'rb'))

    count, best_hr = 0, 0

    for epoch in range(args.epochs):

        model.train()  # Enable dropout (if have).
        start_time = time.time()
        # train_loader.dataset.ng_sample()
        cnt = 0

        for user, item, label in train_loader:
            # print(train_loader)
            # print(user,item)
            user = user.to(device=args.device)
            item = item.to(device=args.device)
            label = label.float().to(device=args.device)
            # print(user, item)
            # label=label.numpy()
            model.zero_grad()
            prediction = model(user, item)
            print(prediction)
            loss = loss_function(prediction, label)
            print(loss)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            cnt += 1

        model.eval()
        HR, NDCG, RECALL, PREC = evaluate.metrics1(epoch, method, model, test_loader, args.top_k, args.device)
        # print(cnt)
        cnt_batch = int((np.shape(train_data)[0] + np.shape(train_neg_data)[0]) / args.batch_size) + 1
        # print(cnt_batch)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'NeuMF.pth'))

        sample_size = args.sample_size
        log_model_prob = np.zeros((user_num, sample_size), float)
        rewards_uid = []
        rewards_list = {}
        ranking_list = {}
        scores={}


        i = 0
        for user, item, label in train_fair_loader:

            rewards = []
            rankings = []

            user = user.to(device=args.device)
            item = item.to(device=args.device)
   
            label = label.numpy()
            scores[i] = model(user, item)
            prediction_ = torch.nn.Softmax(dim=0)(scores[i])
           
            prediction = prediction_.data.numpy().flatten()
            
            for j in range(sample_size):
                ranking = IndividualFairnessLoss.sample_ranking(prediction, False)
                
                rankings.append(ranking)
               
                ndcg, dcg = IndividualFairnessLoss.compute_dcg(ranking, label, 10)
                rewards.append(ndcg)
            rewards_list[i] = rewards
            ranking_list[i] = rankings

            rewards_uid.append(np.mean(rewards_list[i]))
            i += 1

        pdiff = IndividualFairnessLoss.compute_pairwise_disparity_matrix(test_pid_list, ranking_list,
                                                                                     sorted_graph, item_num, user_num,
                                                                                     reachable_path_matrix, args.alpha,
                                                                                     args.beta)

        individual_fairness_coeffs = pdiff

        model.zero_grad()
        for u in range(user_num):
            for j in range(sample_size):
                ranking =  ranking_list[u][j]
                reward = rewards_list[u][j]

                log_model_prob = IndividualFairnessLoss.compute_log_model_probability(args,
                    scores[u], ranking)

                reinforce_loss = args.lambda_reward * log_model_prob * -reward


                individual_fairness_cost = float(args.lambda_ind_fairness *(individual_fairness_coeffs[j])) * log_model_prob

                reinforce_loss += individual_fairness_cost


            reinforce_loss.backward()

        optimizer.step()

        model.eval()
        HR, NDCG, RECALL, PREC = evaluate.metrics2(epoch, method, model, test_loader, args.top_k, args.device)
        print(cnt)
        cnt_batch = int((np.shape(train_data)[0] + np.shape(train_neg_data)[0]) / args.batch_size) + 1
        print(cnt_batch)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model.state_dict(), os.path.join(args.model_path, 'NeuMF.pth'))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
