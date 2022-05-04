import numpy as np
import sys,math
import torch

def sample_ranking(probs, output_propensities=True):
    probs_ = np.array(probs, copy=True)
    ranking = []
    propensity = 1.0
    candidate_set_size = len(probs_)
    print(candidate_set_size)
    #probs_ = np.clip(probs_, 1e-5, 1.0)
    probs_ = probs_
    try:
        ranking = np.random.choice(
            candidate_set_size,
            size=candidate_set_size,
            p=probs_,
            replace=False)
    except ValueError:
        print(probs_,'********************')
        sys.exit(1)
    if output_propensities:
        for i in range(candidate_set_size):
            propensity *= probs_[ranking[i]]
            probs_[ranking[i]] = 0.0
            probs_ = probs_ / probs_.sum()
        return ranking, propensity
    else:
        return ranking


def compute_pairwise_disparity_matrix(test_pid_list,rankings,sorted_graph,num_pids,num_uids,reachable_path_matrix,alpha,beta):

    num_sim_pair=int(alpha*np.shape(sorted_graph)[0])
    sim_pair=sorted_graph[0:num_sim_pair]
    #num_uids=rankings.keys()
    #print(rankings.keys())
    #print(rankings[1])
    # cov_matrix=np.zeros(len(reachable_path_matrix),int)
    # matrix=0
    difs=0
    sample_size = np.shape(rankings[1])[0]
    cov_list=np.zeros((sample_size,num_pids),int)
    matrix=np.zeros(sample_size,float)
    for i in rankings.keys():#number of uids
        ranking=rankings[i]#ranking=[[],[],...,[]_sample_size]
        for j in range(np.shape(ranking)[0]):#j in sample_size
            for pid_index in ranking[j]:
                pid=test_pid_list[i][pid_index]
                cov_list[j][pid]+=1
    gggg=0
    unfair_pair=[]
    for k in range(sample_size):
        for pair in sim_pair:
            i =pair[0]
            j=pair[1]

            M=reachable_path_matrix[i]+reachable_path_matrix[j]
            if M<(cov_list[k][i]+cov_list[k][j]):
                M = cov_list[k][i] + cov_list[k][j]
            dif=abs(cov_list[k][i]-cov_list[k][j])/(M+1e-07)-beta

            if (reachable_path_matrix[i]==0 and reachable_path_matrix[j]==0):
                gggg+=1
                #print('88888888888888888')
                dif=0
            #print(dif,cov_matrix[k][i],cov_matrix[k][j],reachable_path_matrix[i],reachable_path_matrix[j])
            if dif>0:
                matrix[k]+=dif/num_sim_pair
                #print("uhhrn,bmzbhjfutd",gggg)

            # dif = abs(cov_matrix[k][i] - cov_matrix[k][j])
            # matrix[k] += dif
            # #
            # # dif = abs(cov_list[i] - cov_list[j]) / ((
            # #     cov_matrix[k][i] + cov_matrix[k][j]) + 1e-07) - beta
            #
            #
            #
            # if dif > 0:
            #     matrix[k]+=dif
            #     unfair_pair.append([k, i, j, dif])
    return matrix

def compute_pairwise_disparity_matrix217(test_pid_list,rankings,sorted_graph,num_pids,num_uids,reachable_path_matrix,alpha,beta):

    num_sim_pair=int(alpha*np.shape(sorted_graph)[0])
    sim_pair=sorted_graph[0:num_sim_pair]
    #num_uids=rankings.keys()
    #print(rankings.keys())
    #print(rankings[1])
    # cov_matrix=np.zeros(len(reachable_path_matrix),int)
    # matrix=0
    difs=0
    sample_size = np.shape(rankings[1])[0]
    cov_list=np.zeros((sample_size,num_pids),int)
    matrix=np.zeros(sample_size,float)
    for i in rankings.keys():#number of uids
        ranking=rankings[i]#ranking=[[],[],...,[]_sample_size]
        for j in range(np.shape(ranking)[0]):#j in sample_size
            print(ranking[j])#[0:10]means top-10
            for pid_index in ranking[j]:
                pid=test_pid_list[i][pid_index]
                cov_list[j][pid]+=1
    gggg=0
    unfair_pair=[]
    for k in range(sample_size):
        for pair in sim_pair:
            i =pair[0]
            j=pair[1]

            M=reachable_path_matrix[i]+reachable_path_matrix[j]

            dif=abs(cov_list[k][i]-cov_list[k][j])/(M+1e-07)-beta

            if (reachable_path_matrix[i]==0 and reachable_path_matrix[j]==0):
                gggg+=1
                #print('88888888888888888')
                dif=0
            #print(dif,cov_matrix[k][i],cov_matrix[k][j],reachable_path_matrix[i],reachable_path_matrix[j])
            if dif>0:
                matrix[k]+=dif/num_sim_pair
                #print("uhhrn,bmzbhjfutd",gggg)

            # dif = abs(cov_matrix[k][i] - cov_matrix[k][j])
            # matrix[k] += dif
            # #
            # # dif = abs(cov_list[i] - cov_list[j]) / ((
            # #     cov_matrix[k][i] + cov_matrix[k][j]) + 1e-07) - beta
            #
            #
            #
            # if dif > 0:
            #     matrix[k]+=dif
            #     unfair_pair.append([k, i, j, dif])
    return matrix




def compute_pairwise_disparity_matrix1(test_pid_list,rankings,sorted_graph,log_model_prob,num_pids,num_uids,reachable_path_matrix,alpha,beta):

    num_sim_pair=int(alpha*np.shape(sorted_graph)[0])
    sim_pair=sorted_graph[0:num_sim_pair]
    #num_uids=rankings.keys()
    #print(rankings.keys())
    #print(rankings[1])
    # cov_matrix=np.zeros(len(reachable_path_matrix),int)
    # matrix=0
    difs=0
    sample_size = np.shape(rankings[1])[0]
    cov_list=np.zeros((sample_size,num_pids),int)
    matrix=np.zeros(sample_size,float)
    for i in rankings.keys():#number of uids
        ranking=rankings[i]#ranking=[[],[],...,[]_sample_size]
        for j in range(np.shape(ranking)[0]):#j in sample_size
            for pid_index in ranking[j]:
                pid=test_pid_list[i][pid_index]
                cov_list[j][pid]+=log_model_prob[i][j]
    gggg=0
    unfair_pair=[]
    for k in range(sample_size):
        for pair in sim_pair:
            i =pair[0]
            j=pair[1]

            dif=abs(cov_list[k][i]-cov_list[k][j])/(reachable_path_matrix[i]+reachable_path_matrix[j]+1e-07)-beta

            if (reachable_path_matrix[i]==0 and reachable_path_matrix[j]==0):
                gggg+=1
                #print('88888888888888888')
                dif=0
            #print(dif,cov_matrix[k][i],cov_matrix[k][j],reachable_path_matrix[i],reachable_path_matrix[j])
            if dif>0:
                matrix[k]+=dif/num_sim_pair
                #print("uhhrn,bmzbhjfutd",gggg)

            # dif = abs(cov_matrix[k][i] - cov_matrix[k][j])
            # matrix[k] += dif
            # #
            # # dif = abs(cov_list[i] - cov_list[j]) / ((
            # #     cov_matrix[k][i] + cov_matrix[k][j]) + 1e-07) - beta
            #
            #
            #
            # if dif > 0:
            #     matrix[k]+=dif
            #     unfair_pair.append([k, i, j, dif])
    return matrix
def compute_log_model_probability(args,scores, ranking):
    """
    more stable version
    if rel is provided, use it to calculate probability only till
    all the relevant documents are found in the ranking
    """
    subtracts = torch.zeros_like(scores).to(device=args.device)
    log_probs = torch.zeros_like(scores).to(device=args.device)
    # if gpu_id is not None:
    #     subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs],
    #                                                gpu_id)
    for j in range(len(ranking)):
        posj = ranking[j]
        log_probs[j] = scores[posj] - logsumexp(scores - subtracts, dim=0)
        subtracts[posj] = scores[posj] + 1e6
    return torch.sum(log_probs)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs



def compute_dcg(ranking, relevance_vector, k=10000):
    """
    returns the array. actual dcg is the sum or average of this array
    """
    dcgmax = 0.0
    sorted_relevances = -np.sort(-relevance_vector)
    N = len(relevance_vector)
    if k == 0:
        k = N
    for i, relevance in enumerate(sorted_relevances[:min((k, N))]):
        dcgmax += float(2.0**relevance - 1) / math.log2(2 + i)
    dcg = 0.0
    for i, doc in enumerate(ranking[:min((k, N))]):
        dcg += float(2**relevance_vector[doc] - 1) / math.log2(2 + i)
    if dcgmax == 0:
        # print(relevance_vector, ranking)
        return 1.0, 1.0
    else:
        return dcg / dcgmax, dcg


def compute_pairwise_disparity_matrix223(test_pid_list,rankings,sorted_graph,num_pids,num_uids,reachable_path_matrix,alpha,beta):

    sample_size = np.shape(rankings[1])[0]
    cov_list=np.zeros((sample_size,num_pids),int)
    matrix=np.zeros(sample_size,float)
    for i in rankings.keys():#number of uids
        ranking=rankings[i]#ranking=[[],[],...,[]_sample_size]
        for j in range(np.shape(ranking)[0]):#j in sample_size
            for pid_index in ranking[j]:
                pid=test_pid_list[i][pid_index]
                cov_list[j][pid]+=1
    gggg=0
    unfair_pair=[]

    for k in range(sample_size):
        for pair in sorted_graph:
            i =pair[0]
            j=pair[1]
            sim=pair[2]

            M=reachable_path_matrix[i]+reachable_path_matrix[j]
            if M<(cov_list[k][i]+cov_list[k][j]):
                M = cov_list[k][i] + cov_list[k][j]

            dif=abs(cov_list[k][i]-cov_list[k][j])/(M+1e-07)-beta/sim

            if (reachable_path_matrix[i]==0 and reachable_path_matrix[j]==0):
                gggg+=1
                #print('88888888888888888')
                dif=0
            #print(dif,cov_matrix[k][i],cov_matrix[k][j],reachable_path_matrix[i],reachable_path_matrix[j])
            if dif>0:
                matrix[k]+=dif/np.shape(sorted_graph)[0]
                #print("uhhrn,bmzbhjfutd",gggg)

            # dif = abs(cov_matrix[k][i] - cov_matrix[k][j])
            # matrix[k] += dif
            # #
            # # dif = abs(cov_list[i] - cov_list[j]) / ((
            # #     cov_matrix[k][i] + cov_matrix[k][j]) + 1e-07) - beta
            #
            #
            #
            # if dif > 0:
            #     matrix[k]+=dif
            #     unfair_pair.append([k, i, j, dif])
    return matrix
