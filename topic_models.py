"""A python module that contains code for training and saving a topic model.

"""
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from defense.lda import LDA
from defense.fdptm_helpers import sanitize_corpus
from data.utils import split_data, extract_features

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

def train_lda_gibbs(X, k, n_iters, doc_topic_prior = None, topic_word_prior = None):
    """
    A function that trains LDA with k topics on the data X using our gibbs
    sampling implementation.

    Args:
    - X (np.ndarray): The data represented as a document-term matrix that is
        (num documents x vocab length) in size.
    - k (int): The number of topics assumed for LDA
    - doc_topic_prior (float): Dirichlet prior parameter alpha in Blei et al 
    - topic_word_prior (float): Dirichlet prior parameter beta in Blei et al 
    - n_iters (int): The number of training iterations to run gibbs sampling.

    Returns:
        - phi: The topic word distribution as a (k x vocab length) shape matrix
    """
    if doc_topic_prior is None:
        doc_topic_prior = 1/k

    if topic_word_prior is None:
        topic_word_prior = 1/k

    lda = LDA(X, k, doc_topic_prior, topic_word_prior)
    lda.simple_fit(n_iters)

    return lda.phi

def train_dp_lda_gibbs(X, k, epsilon, n_iters, doc_topic_prior = None, 
                    topic_word_prior = None):
    """
    A function that trains LDA with k topics on the data X using our dp gibbs
    sampling implementation from Zhu et al (2016).

    Args:
    - X (np.ndarray): The data represented as a document-term matrix that is
        (num documents x vocab length) in size.
    - k (int): The number of topics assumed for LDA
    - doc_topic_prior (float): Dirichlet prior parameter alpha in Blei et al 
    - topic_word_prior (float): Dirichlet prior parameter beta in Blei et al 
    - n_iters (int): The number of training iterations to run gibbs sampling.

    Returns:
        - phi (ndarray): The topic word distribution as a (k x vocab length) shape matrix
        - perplexity (float): The perplexity is an indicator of model fit!
    """
    if doc_topic_prior is None:
            doc_topic_prior = 1/k
    
    if topic_word_prior is None:
        topic_word_prior = 1/k
        
    lda = LDA(X, k, doc_topic_prior, topic_word_prior)
    lda.dl_private_fit(n_iters, epsilon)

    return lda.phi

def fdptm(text, dpsu_func, dpsu_kwargs, train_func, train_kwargs, p = 1):
    """
    A function that runs the Fully Differentially Private Topic Modeling
    algorithm of a p proportion of the provided data.

    Args:
        - text (list[str]): List of str documents.
        - dpsu_func (function()): The vocabulary selection algorithm.
            if None, then it skips the DP-vocabulary selection step.
            (i.e. choose_vocab from defense.fdptm_helpers is provided choice)
        - dpsu_kwargs (Dict): Key word arguments of vocabulary selection algorithm
        - train_func (function()): A function that returns the topic-word dist.
            for all documents in selected for the target model. (see above)
        - train_kwargs (Dict): The key word arguments for the train function.
        - p (float): Percentage of the data to include in the model (default 1)

    Returns:
        - docs_in_train (ndarray): Documents included
        - vocab (ndarray):
        - phi (ndarray):
    """
    docs_in_train = split_data(text, p)
    train_docs = np.array(text)[docs_in_train].tolist()

    if dpsu_func:
        private_vocab = dpsu_func(train_docs, **dpsu_kwargs)
        train_docs = sanitize_corpus(private_vocab, train_docs)

    X, vocab = extract_features(train_docs)

    target_phi = train_func(X, **train_kwargs)
    
    return docs_in_train, vocab, target_phi

if __name__ == "__main__":
    # Here is a simple simulation showing the FDPTM in action!
    print("hello")