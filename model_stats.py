"""
Python module that contains various functions for calculating model statistics.

"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.special import logit

def max_lda_logll(phi, doc_encoded):
    """
    This estimates max_q log(doc | q, phi). We treat q as a fixed distribution over
    topics and maximize over it. It's a heuristic for the max logll of the doc given
    the model under LDA.
    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - (float) max_q log(doc | q, phi
    """
    theta = max_topic_dist(phi, doc_encoded)
    return neg_logll(theta, phi, doc_encoded)

def all_lda_stats(phi, doc_encoded):
    """
    Calculates the max log-likelihood, the entropy, the standard deviation, and 
    the logit maximum of the topic distribution that maximizes document log-likelihood.

    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - list(float) [max_q log(doc | q, phi), entropy(q)]
    """
    theta = max_topic_dist(phi, doc_encoded)

    logll = neg_logll(theta, phi, doc_encoded)
    entro = -1 * entropy(theta) #switch the direction of the entropy for offline
    max_p = logit(max(theta)-10e-12) # Round down from 1 
    std = np.std(theta)

    return [logll, entro, max_p, std]

def max_dmm_logll(phi, doc_encoded):
    """
    This estimates max_z log(doc | z, phi). We treat z as a single topic. It's a 
    heuristic for the max logll of the doc given the model under DMM.
    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - (float) max_z log(doc | q, phi
    """
    return np.log(phi[:, doc_encoded]).sum(axis = 1).max()

def simple_log_sum(phi, doc_encoded):
    """
    This function neglects the generative process for the topic model and calculates
    a giant log sum of all word probabilities for all topics for the document.
    """
    return np.log(phi[:, doc_encoded]).sum()

def max_topic_dist(phi, doc_encoded):
    """
    Calculates the topic (theta) distribution that maximizes the log-likelihood
    of the document.
    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - theta (ndarray): phi.shape[0]-dimensional topic distribution for
            the document.
    """
    k = phi.shape[0]
    init = np.repeat(1/k, k)
    bound = [(0.0, 1.0) for _ in range(k)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    out = minimize(neg_logll, init, args=(phi, doc_encoded), bounds = bound, constraints=cons)
    theta = out['x']
    return theta

def neg_logll(q, phi, doc_encoded):
    """
    This is an estimate of -log p(doc | q, phi) for any q under LDA.
    Args:
        - q (np.ndarray): a document-topic distribution with shape (num_topics,)
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - (float): -log p(doc | q, phi)
    """
    p = phi[:,doc_encoded]
    return -np.sum(np.log(np.sum((p.T * q).T, axis = 0)))

def get_topic_dist(phi, doc_encoded, tol=10e-3, max_iter=15):
    """
    Estimate the document topic distribution through Gibbs sampling.
    Not a full implementation of folding-in, because we start with random topic assignments
    for each word. Uses assumption of LDA.
    Args:
        phi (np.array): the topic-word distribution associated with a TM.
        doc_encoded (np.ndarray): document represented as array of indices.
        tol (float): tolerance for convergence. 
            Function terminates if changes for each topic p are less than tol
        max_iter (int): maximum number of iterations through the document. 
            15 tends to be enough, but we can always boost it for long docs.

    Returns:
        theta (np.array): the document-topic distribution
    """
    # Initialize topic assignments and count vector for each sampled word
    z_d = np.random.choice(phi.shape[0], size=len(doc_encoded))
    counts = np.bincount(z_d, minlength=phi.shape[0]) + 1 / phi.shape[0] # smoothing parameter
    theta = counts/counts.sum()
    
    it = 0
    while it < max_iter:
        for idx, sample_w_idx in enumerate(doc_encoded):
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
