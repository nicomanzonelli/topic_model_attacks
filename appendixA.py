#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import norm
from scipy.stats import entropy
from scipy.optimize import minimize

from .attack_simulations import split_and_bin
from .utils import train_model_sklearn, calc_max_logll, prob_f

#For the KL Divergence stuff
import warnings
warnings.filterwarnings("ignore")

"""
Functions for setting up the experement
"""

def train_and_store(splits, k, verbose = True):
    """
    Input:
        - splits (list of 2D-arrays): Random subset of X with half the observations.
            Output of split_and_bin.
        - k (int): Number of topics to fit with LDA.
    Return:
        - phis (list): LDA model Loglikelihood estimation.
    """
    
    # Train LDA on each split
    phis = []
    if verbose:
        for split in tqdm(splits, disable = (not verbose)):
            phi.append(train_model_sklearn(split, k = k))
        
    return phis

def statistics_across_phis(X, phis, docs_in_split):
    """
    Functions to calculate our query statistics across each phi and split.
    Automatically creates a nice dictionary where
    """
    d = {'doc_idx': [], 
         'in_cond_logll': [], 'out_cond_logll': [], 
         'in_max_logll': [], 'out_max_logll': [],
         'in_theta': [], 'out_theta': [],
         'in_logll': [], 'out_logll': [],
         'in_perp': [], 'out_perp': [],
         'in_std': [], 'out_std': [],
         'in_max': [], 'out_max': [],
         'in_entropy': [], 'out_entropy': []
        }
    
    for doc_idx in tqdm(range(X.shape[0])):
        # Define more storage
        d['doc_idx'].append(doc_idx)
        dsub = {t:[] for t in list(d.keys())[1:]}
        
        # query Statistcs on phi
        doc_w_idx = get_doc_word_idx(X[doc_idx,:])
        for split_idx in range(len(phis)):
            if doc_idx in docs_in_split[split_idx]:
                dsub['in_cond_logll'].append(calc_cond_statistic(phis[split_idx], doc_w_idx))
                dsub['in_max_logll'].append(calc_max_logll(phis[split_idx], doc_w_idx))
                theta = est_doc_topic_dist(phis[split_idx], doc_w_idx)
                logll = calc_logll(phis[split_idx], theta, doc_w_idx)
                perp = calc_perplexity(logll, doc_w_idx)
                dsub['in_theta'].append(theta)
                dsub['in_logll'].append(logll)
                dsub['in_perp'].append(perp)
                dsub['in_std'].append(np.std(theta))
                dsub['in_max'].append(np.max(theta))
                dsub['in_entropy'].append(calc_entropy(theta))
                
            else:
                dsub['out_cond_logll'].append(calc_cond_statistic(phis[split_idx], doc_w_idx))
                dsub['out_max_logll'].append(calc_max_logll(phis[split_idx], doc_w_idx))
                theta = est_doc_topic_dist(phis[split_idx], doc_w_idx)
                logll = calc_logll(phis[split_idx], theta, doc_w_idx)
                perp = calc_perplexity(logll, doc_w_idx)
                dsub['out_theta'].append(theta)
                dsub['out_logll'].append(logll)
                dsub['out_perp'].append(perp)
                dsub['out_std'].append(np.std(theta))
                dsub['out_max'].append(np.max(theta))
                dsub['out_entropy'].append(calc_entropy(theta))
                
        # Add back to big dict
        for t in list(dsub.keys()):
            d[t].append(dsub[t])
        
    return pd.DataFrame(d)

"""
Functions for estimating our statistics of intererst
"""

def get_doc_word_idx(target_document):
    """
    Represent the document as an array of word indicies where word ordering doesnt matter.
    Input:
        - target_document (array): Document represented as vector of word counts.
    Output:
        - target_doc_word_idx (array): Document represented as array of indicies.
    """
    target_doc_word_idx = []
    for word_idx in np.where(target_document>0)[0]:
        target_doc_word_idx += [word_idx] * target_document[word_idx]
    return target_doc_word_idx

def est_doc_topic_dist(phi, doc_word_idx, tol=1e-6, max_iter=10):
    """
    Estimate the document topic distribution through Gibbs sampling or query sampling.
    Not a full implementation of folding-in, because we start with random topic assignments
    for each word. Uses assumption of LDA.
    Inputs:
        phi (np.array): the topic-word distribution associated with a TM.
        doc_word_idx (np.array): document represented as array of indices.
        tol (float): tolerance for convergence. 
            Function terminates if changes for each topic p are less than tol
        max_iter (int): maximum number of iterations through the document
        w_samples (int): the number of samples to draw from the document
        
    Outputs:
        theta (np.array): the document-topic distribution
    """
    # Sample words from the document (helps for really short or really long documents)
    # sample = np.random.choice(doc_word_idx, size=w_samples, replace=True)
    
    # Initialize topic assignments and count vector for each sampled word
    z_d = np.random.choice(phi.shape[0], size=len(doc_word_idx))
    counts = np.bincount(z_d, minlength=phi.shape[0]) + 1 / phi.shape[0] # smoothing parameter
    theta = counts/counts.sum()
    
    it = 0
    while it < max_iter:
        np.random.shuffle(doc_word_idx)
        for idx, sample_w_idx in enumerate(doc_word_idx):
            # Decrease count for the assigned word's topic-assignment
            counts[z_d[idx]] -= 1     
            # Compute the probability of that word being assigned to each topic
            p_z_t = phi[:, sample_w_idx] * counts
            p_z_t /= np.sum(p_z_t)         
            # Sample a new topic
            topic_sample = np.random.choice(phi.shape[0], p=p_z_t) 
            # Update the topic assignment and the counts
            z_d[idx] = topic_sample
            counts[z_d[idx]] += 1
        # Check for convergence
        new_theta = counts / counts.sum()
        diff = np.abs(new_theta - theta)
        if np.all(diff < tol):
            break
        # Update theta and continue iteration
        theta = new_theta
        it += 1
    return theta

def calc_cond_statistic(phi, doc_word_idx):
    """
    This estimates max_q log(doc | q, phi). We treat q as a fixed distribution over
    topics and maximize over it. It's a hueristic for the max logll of the doc given
    the model under LDA.
    Inputs:
        - phi 
        - doc_word_idx
    Outputs:
        - (float) max_q log(doc | q, phi
    """
    return np.log(phi[:, doc_word_idx]).sum(axis = 1).max()

def calc_logll(phi, theta, doc_word_idx):
    """
    This estimates log(doc | q, phi). We treat q as a distribution over topics. 
    If we calculate theta with query sampling/likelihood then it's
    the model under LDA.
    Inputs:
        - phi 
        - doc_word_idx
    Outputs:
        - (float)log(doc | theta, phi)
    """
    return -prob_f(theta, phi, doc_word_idx)

def calc_perplexity(logll, doc_word_idx):
    """
    This estimates perplexity. We treat q as a distribution over topics. 
    If we calculate theta with query sampling/likelihood then it's
    the model under LDA.
    Inputs:
        - logll (float): the estimated log-likelihood
    Outputs:
        - (float) perplexity
    """
    return np.exp(-logll/len(doc_word_idx))

def calc_entropy(theta):
    """
    Function that gets the entropy of the estimated topic-word distribution
    """
    return entropy(theta)

"""
Functions for calculating KL divergence
"""

def calc_kl(in_stat, out_stat):
    """
    Estimates a normal and calculates the kl divergence between the two distributions.
    """
    with_norm = norm(loc = np.mean(np.array(in_stat)), scale = np.std(np.array(in_stat)))
    without_norm = norm(loc = np.mean(np.array(out_stat)), scale = np.std(np.array(out_stat)))
    
    max_std = max(np.std(np.array(in_stat)),  np.std(np.array(out_stat)))
    max_p = max(np.mean(np.array(in_stat)), np.mean(np.array(out_stat))) + 3*max_std
    min_p = min(np.mean(np.array(in_stat)), np.mean(np.array(out_stat))) - 3*max_std
    points_to_eval = np.arange(min_p, max_p, .25)
    
    with_pdf = with_norm.pdf(points_to_eval)
    without_pdf = without_norm.pdf(points_to_eval)
    
    return entropy(with_pdf, without_pdf)

"""
Functions for plotting
"""
def plot_statistic_quad(stats, sup_title, names_idx, statistic_tup, save=False):
    fig, axs = plt.subplots(2,2, figsize = (7,7))
    axs = axs.flatten()

    for idx, (name, doc_idx) in enumerate(names_idx.items()):

        axs[idx].hist(stats[stats['doc_idx'] == doc_idx][statistic_tup[0]], label = "with document", alpha = .3)
        axs[idx].hist(stats[stats['doc_idx'] == doc_idx][statistic_tup[1]], label = "without document", alpha = .3)
        axs[idx].set_title(f"{name}")

    plt.legend()
    fig.suptitle(sup_title)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(sup_title + '.png')
        
if __name__ == "__main__":
    from utils import read_json, write_pickle, create_doc_term_mat
    # Read in raw dataset
    data = read_json('/Users/nico/Documents/thesis/topic_model_attacks/data/newsgroup_clean.json', 'r')
        
    # Create document-term matrix
    raw_corpus = data['text']
    X, _ = create_doc_term_mat(raw_corpus)
    
    # This is the code to run our simple experiment
    print("Creating Splits...")
    splits, docs_in_split = split_and_bin(X, 200)
    print("Training LDA on Each Split...")
    phis = train_and_store(splits, 20)
    print("Computing Statistics on Phi...")
    stats = statistics_across_phis(X, phis, docs_in_split)

    with open('sim_results_news.pickle', 'wb') as f:
        pickle.dump(stats, f)