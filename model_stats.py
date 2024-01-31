"""
Python module that contains various functions for calculating model statistics.

"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.special import logit

def lda_stats(phi, doc_encoded):
    """
    Calculates the max log-likelihood and the entropy of the topic distribution that
    maximizes document log-likelihood.
    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - list(float) [max_q log(doc | q, phi), entropy(q)]
    """
    k = phi.shape[0]
    init = np.repeat(1/k, k)
    bound = [(0.0, 1.0) for _ in range(k)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    out = minimize(neg_logll, init, args=(phi, doc_encoded), bounds = bound, constraints=cons)

    logll = -out['fun']
    entro = -1 * entropy(out['x']) #switch the direction of the entropy for offline
    max_p = logit(max(out['x'])-10e-12) # Round down from 1 
    std = np.std(out['x'])

    return [logll, entro, max_p, std]

def dmm_stats(phi, doc_encoded):
    """
    Calculates the max log-likelihood and the entropy of the document's topic 
    dist under the DMM assumptions (document's are generated by one document).
    """
    logll = max_dmm_logll(phi, doc_encoded)
    entro = dmm_topic_entropy(phi, doc_encoded)
    max_p = dmm_max_p_logit(phi, doc_encoded)

    return [logll, entro, max_p]

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
    k = phi.shape[0]
    init = np.repeat(1/k, k)
    bound = [(0.0, 1.0) for _ in range(k)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    out = minimize(neg_logll, init, args=(phi, doc_encoded), bounds = bound, constraints=cons)
    return -out['fun']

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

def lda_topic_entropy(phi, doc_encoded):
    """
    This estimates max_q log(doc | q, phi). We treat q as a fixed distribution over
    topics and maximize over it. Now q is the topic distribution that maximizes
    the document likelihood. We take the entropy of (q).
    Args:
        - phi (np.ndarray): the topic-word distribution associated with a TM.
        - doc_encoded (np.ndarray): document represented as array of indices.
    Returns:
        - (float) entropy(q) | max_q log(doc | q, phi)
    """
    k = phi.shape[0]
    init = np.repeat(1/k, k)
    bound = [(0.0, 1.0) for _ in range(k)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    out = minimize(neg_logll, init, args=(phi, doc_encoded), bounds = bound, constraints=cons)
    return -1 * entropy(out['x'])

def dmm_topic_entropy(phi, doc_encoded):
    """
    This function estimates the entropy of the topic distribution for the given
    document under the assumption that the document is generated by one document
    like in the DMM.
    """
    return -1 * entropy((np.log(phi[:, doc_encoded]).sum(axis = 1))+10e-32)

def dmm_max_p_logit(phi, doc_encoded):
    """
    Takes the maximum probability of the topic distribution of the given document
    under the generative assumptions of DMM and applies logit transformation to
    help enforce normality.
    """
    logit(np.log(phi[:, doc_encoded]).sum(axis =1))
    

def simple_log_sum(phi, doc_encoded):
    """
    This function neglects the generative process for the topic model and calculates
    a giant log sum of all word probabilities for all topics for the document.
    """
    return np.log(phi[:, doc_encoded]).sum()