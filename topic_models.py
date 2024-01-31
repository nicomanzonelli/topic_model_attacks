"""A python module that contains code for training and saving a topic model.

"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from dp_defense.lda import LDA

def train_lda_sklearn(X, k, doc_topic_prior = None, topic_word_prior = None):
    """
    A function that trains a LDA with k topics on the data X.

    Args:
        - X (np.ndarray): The data represented as a document-term matrix that is
            (num documents x vocab length) in size.
        - k (int): The number of topics assumed for LDA
        - doc_topic_prior (float): Dirichlet prior parameter alpha in Blei et al 
        - topic_word_prior (float): Dirichlet prior parameter beta in Blei et al 

    Returns:
        - phi: The topic word distribution as a (k x vocab length) shape matrix
    """
    lda = LatentDirichletAllocation(n_components=k, 
                                    doc_topic_prior = doc_topic_prior, 
                                    topic_word_prior = topic_word_prior)
    lda.fit(X)
    phi = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    return phi

def train_dp_lda_gibbs(X, k, epsilon, doc_topic_prior, topic_word_prior, n_iters):
    """
    A function that trains LDA with k topics on the data X using the DP gibbs
    sampling implementation from PAPER.

    Args:
    - X (np.ndarray): The data represented as a document-term matrix that is
        (num documents x vocab length) in size.
    - k (int): The number of topics assumed for LDA
    - epsilon (float): The privacy loss parameter.
    - doc_topic_prior (float): Dirichlet prior parameter alpha in Blei et al 
    - topic_word_prior (float): Dirichlet prior parameter beta in Blei et al 
    - n_iters (int): The number of training iterations to run dp gibbs sampling.

    Returns:
        - phi: The topic word distribution as a (k x vocab length) shape matrix
    """
    if alpha == None:
        alpha = 1/k
    if beta == None:
        beta = 1/k
        
    lda = LDA(X, k, alpha, beta)
    lda.dl_private_fit(n_iters, epsilon)
    return lda.phi
