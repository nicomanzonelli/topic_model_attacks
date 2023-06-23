import json
import pickle
import numpy as np

from scipy.optimize import minimize
from scipy.stats import entropy

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

"""
Simple functions to haldle data
"""
def read_json(in_path):
    with open(in_path, 'r') as f:
        data = json.load(f)
    return data

def write_pickle(obj, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(obj, f)

def create_doc_term_mat(raw_corpus):
    vectorizor = CountVectorizer()
    X = vectorizor.fit_transform(raw_corpus).toarray()
    idx2word = vectorizor.get_feature_names_out()
    return X, idx2word

def sample_documents(X, p):
    """
    A function to sample documments
        - X (array): Corpus represented as the document word count matrix
        - p (float): Binomial parameter which controls which proportion of documents
            to include in the trainset
    Outputs:
        - train_documents (2d-array): Sampled corpus of documents
        - train_document_idx (array): Array of the document indicies included in the trainset.
    """
    # Make binary vector from the binomial (1 indicates yes!)
    attack_docs = np.random.binomial(size = X.shape[0], n = 1, p = p)
    attack_docs_idx = np.where(attack_docs == 1)[0]
    return X[attack_docs_idx,:], attack_docs_idx

"""
Functions to calculate our statistic on phi
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

def prob_f(q, phi, doc_word_idx):
    """
    This is an estimate of -log p(doc | q, phi) for any q under LDA.
    Inputs:
        - q (np.array): a document-topic distribution
        - phi (np.array): the topic-word distribution associated with a TM.
        - doc_word_idx (np.array): document represented as array of indicies.
    Outputs:
        - (float): -log p(doc | q, phi)

    """
    p = phi[:,doc_word_idx]
    return -np.sum(np.log(np.sum((p.T * q).T, axis = 0)))

def calc_max_logll(phi, doc_word_idx):
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
    k = phi.shape[0]
    init = np.repeat(1/k, k)
    bound = [(0.0, 1.0) for _ in range(k)]
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    out = minimize(prob_f, init, args=(phi ,doc_word_idx), bounds = bound, constraints=cons)
    return -out['fun']

def calc_cond_statistic(phi, doc_word_idx):
    """
    This estimates max_q log(doc | q, phi). We treat q as a single topic and maximize over it. 
    It's a hueristic for the max logll of the doc under a simple generative process,
    like the Dirichlet Mixture Model (DMM) where every document belongs to 1 topic.
    Inputs:
        - phi 
        - doc_word_idx
    Outputs:
        - (float) max_q log(doc | q, phi
    """
    return np.log(phi[:, doc_word_idx]).sum(axis = 1).max()


"""
Functions to help replicate Huang et. al (2022) attacks against topic models.
"""

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


def calc_entropy(theta):
    """
    Function that gets the entropy of the estimated topic-word distribution
    """
    return entropy(theta)

"""
Functions for learning topic models
"""
def train_model_sklearn(X, k, alpha = None, beta = None, training_iterations=10):
    """
    Function to train a model on a dataset X.
    Inputs:
        - X (array): Corpus in document-term matrix representation
        - alpha (int): Dirichlet prior for doc-topic dist. Default to sklearn default.
        - beta (int): Dirichlet prior topic-word dist. Default to sklearn default.
        - training_iterations (int): VI interations. Default to sklearn default.
    Outputs:
        - attack_phi (array): topic-word distribution of trained LDA
    """
    lda = LatentDirichletAllocation(n_components= k, doc_topic_prior = alpha,
                                   topic_word_prior = beta, max_iter = training_iterations)
    lda.fit(X)
    phi = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    return phi

"""
Functions to assist in experiment evaluation
"""

def calc_tpr(fpr_fixed, fpr, tpr):
    """
    Calculates the true positive rate at a fixed postive rate.
    Attempts to use the tpr that corresponds to an exact fpr as indexed in the arrays, but 
    typically yeilds a linear approx. of the tpr at the fpr_set based on the two closest fprs.
    Input: 
        - fpr_set (float): Fixed false postive rate to estimate the true postive rate at.
        - fpr (array): False positive rates with indicies corresponding to tpr and threshold.
        - tpr (array) True postive rates with indicies corresponding to fpr and thresholds.
    Output:
        - (float): The true postive rate at a fixed false positive rate.
    """
    last_idx = np.where(fpr_fixed >= fpr)[0][-1]
    return tpr[last_idx]