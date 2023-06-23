"""
This module provides the functions for executing LiRA attack simulations on topic models
"""
from offline_LiRA import LiRA_offline
from online_LiRA import LiRA_online
from utils import sample_documents


from tqdm import tqdm
import numpy as np

"""
Functions that set up the LiRA attack
"""

def split_and_bin(X, N):
    """
    Input:
        - X (array): The document-term matrix with shape (# of Docs, Length of Vocab).
        - N (int): The number of times to randomly split the data into two subsets.
    Outputs:
        - splits (list of 2D-arrays): list of subset of X with half the observations.
        - target_in_splits (list of array): list of document indicies in each split.
    """
    splits = []
    docs_idx_in_split = []
    for _ in range(N//2):
        # get idx of docs selected
        docs_idx_in = np.random.choice(X.shape[0], int(X.shape[0]/2), replace=False)
        docs_idx_out = np.setdiff1d(np.arange(X.shape[0]), docs_idx_in)
        
        splits.append(X[docs_idx_in])
        splits.append(X[docs_idx_out])
        docs_idx_in_split.append(docs_idx_in)
        docs_idx_in_split.append(docs_idx_out)

    return splits, docs_idx_in_split
        
def train_shadow_models(X, N, train_func, train_args, verbose = True):
    """
    Function that trains N shadow topic models.
    
    """
    d = {'doc_idx': [], 'in_phis': [], 'out_phis': []}
    
    # We split the training data N times. Each split will be responsible for 1 Shadow Model.
    training_data, training_docs_idx = split_and_bin(X, N)
    
    # Train LDA on each split (these will be the shadow models)
    phis = []
    print("Training Shadow Models...") if verbose else None
    for data in tqdm(training_data, disable = (not verbose)):
        train_args['X'] = data
        phis.append(train_func(**train_args))
        
    # Determine which documents are in each split
    for doc_idx in range(X.shape[0]):
        d['doc_idx'].append(doc_idx)
        shadow_phis = {'in_phis': [], 'out_phis': []}
        for split_idx in range(len(phis)):
            if doc_idx in training_docs_idx[split_idx]:
                shadow_phis['in_phis'].append(phis[split_idx])
            else:
                shadow_phis['out_phis'].append(phis[split_idx])

        d["out_phis"].append(shadow_phis["out_phis"])
        d["in_phis"].append(shadow_phis["in_phis"])

    return d

"""
Functions that run LiRA attack simulations
"""

def LiRA_exp(X, train_sample_p, N, 
                  attack_training_function, attack_params,
                  shadow_training_function, shadow_params, stat_func,
                  verbose = True):
    """
    A funcion to run the basic experiment.
    Inputs:
        - X (np.array): document-term count matrix of a data set
        - N (int): Parameter to control number of shadow models to train.
            For the online test $N$ is the total number of shadow models trained.
        - train_sample_p (float): percentage of X to sample for training data
        - attack_training_function (funcion): python function defined to take a 
            dataset 'X' and other parameters and returns topic-word dist as array.
        - attack_params (dict): key word args for attack_training_function
        - shadow_training_function (function): like attack training function, but
            used to train shadow models.
        - shadow_params (dict): key word args for shadow_training_function
        - stat_func (function): function that calculates a stastistic on phi
        - verbose (bool): controls printing to console and progress bars
    Outputs:
        - sim_out (dict): dictionary containing the following items:
            - attack_phi (array): The attack models topic-word distribution
            - target_truth (array): Binary vector indicating which doc_idx in trainset
            - offline_scores (list): Test statistics from the offline test
            - online_scores (list): Test statistics from the online test
            - stats_in (list): Nested lists of statstic calculated on in shadow models
            - stats_out (list): Nested lists of statstic calculated on out shadow models
            - stat_obs (list): Nested lists of statstic calculated on attack model
    """
    # Choose attack model training data
    X_train, train_docs_idx = sample_documents(X, .5)
    attack_params['X'] = X_train

    # Train Attack Model
    attack_phi = attack_training_function(**attack_params)
    
    # Truth vector of doc idx in training data (0)
    target_truth = 1*np.in1d(np.arange(X.shape[0]), train_docs_idx)
    
    # Split the auxilary set in half many times to train shadow models (should be N/2)
    sp = train_shadow_models(X, N, shadow_training_function, shadow_params, verbose)
        
    # Conduct the online test
    print("Running online test...") if verbose else None
    score_on = []
    score_off = []
    stat_in = []
    stat_out = []
    stat_obs = []
    for doc_idx in tqdm(range(X.shape[0]), disable = (not verbose)):
        lira = LiRA_online(attack_phi, sp['in_phis'][doc_idx], sp['out_phis'][doc_idx], X[doc_idx])
        score_on.append(lira.fit(stat_func, offline = True))
        score_off.append(lira.lambda_offline)
        stat_in.append(lira.in_stats)
        stat_out.append(lira.out_stats)
        stat_obs.append(lira.obs_stats)

    sim_out = {"attack_phi": attack_phi,
              "target_truth": target_truth,
               "online_scores": score_on,
               "score_off": score_off,
               "stats_in": stat_in,
               "stats_out": stat_out,
               "stat_obs": stat_obs
              }
    
    return sim_out
    