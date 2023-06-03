from .lda import LDA
from .dpsu_gw import run_gw
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm

def read_json(in_path):
    with open(in_path, 'r') as f:
        data = json.load(f)
    return data

def create_doc_term_mat(raw_corpus):
    vectorizor = CountVectorizer()
    X = vectorizor.fit_transform(raw_corpus).toarray()
    idx2word = vectorizor.get_feature_names_out()
    return X, idx2word


def sanitize_corpus(vocab, corpus):
    """
    Function that ensures only words in a vocabulary set occur in the corpus
    Input
        - vocab (list): List of tokens which is the vocabulary set
        - corpus (list): List of str of documents
    Output
        - sanitized_docs (list): List of str of sanitized documents
    """
    documents = [c.split(' ') for c in corpus]
    filtered_documents = []
    for document in documents:
        filtered_document = [word for word in document if word in vocab]
        filtered_documents.append(filtered_document)
        
    return [' '.join(d) for d in filtered_documents]


def choose_vocab(corpus, alpha_cutoff, epsilon, delta):
    """
    Function that chooses a vocabulary set in a dp manner
    Inputs: 
        - corpus (list): List of str of documents
        - alpha_cutoff (float/int):
        - epsilon (float/int): dp epsilon parameter
        - delta (float): dp delta parameter
    Outputs:
        - vocab (list): List of tokens which is the vocabulary set
    """
    input_df = [c.split(' ') for c in corpus]
    vocab = run_gw(input_df, alpha_cutoff, epsilon, delta, 'ci', None)
    return vocab

def train_dp_gibbs(X, k, iterations, epsilon, alpha = None, beta = None):
    """
    Function to train DP variant of LDA. 
    Inputs
        - X (array): document-term count matrix
        - k (int): number of topics
        - iterations (int): number of training iterations
        - epsilon (float/int): dp privacy parameter
        - delta (float): dp privacy parameter
        - alpha (float): learning parameter for gibbs (topic-word)
        - beta (float): learning parameter for gibbs (doc-topic)
    Outputs:
        - phi (array): the topic-word distirbution
    """
    if alpha == None:
        alpha = 1/k
    if beta == None:
        beta = 1/k
        
    mod = LDA(X, k, alpha, beta)
    mod.dl_private_fit(iterations, epsilon)
    logll, perp = mod._compute_log_liklihood_perplexity()
    return mod.phi, logll, perp

def get_topic_coherence(X, phi, topic_idx, n, smoothing_param = 1):
    """
    This function computes the topic coherence score for the specified topic
    This is the UMASS Coherence score from Mimno et al. (2011)
    Inputs:
       - topic_idx (int): topic index to calculate coherence scores for
       - n (int): n-terms to get coherence score for
       - smoothing_param (int/float): Default is 1 which is reccomended in Mimno et al.
           Ensures we do not take the log of 0.
    Outputs:
        - topic_coherence (float): topic coherence for
    """
    #get top n word indexs
    word_idxs = (-phi[topic_idx,:]).argsort()[:n]

    logs = []

    #Loop through word pairs
    for vj in range(1,len(word_idxs)):
        # check how many documents word vj is used in all documents (denom)
        word_j = word_idxs[vj]
        Dvj = np.sum(X[:,word_j] > 0)

        # To handle words in the vocabulary that happen to be top n in topic
        if Dvj != 0:
            for vi in range(vj):
                #check word co-occurances (numerator)
                word_i = word_idxs[vi]
                Dvivj = np.sum((X[:,word_i] > 0) * (X[:,word_j] > 0))

                # calculate pair ratio
                logs.append(np.log((Dvivj+smoothing_param)/Dvj))

    #sum across all pairs
    topic_coherence = np.sum(logs)

    return topic_coherence

def top_n_words(phi, topic, n, vocabulary):
    top_n_words = {}
    
    idxs = (-phi[topic,:]).argsort()[:n]
    for idx in idxs:
        top_n_words[vocabulary[idx]] = phi[topic,:][idx]
        
    return top_n_words

def run_fdptm(corpus, dpsu_func, dpsu_func_params, dp_train_func, dp_train_params):
    """
    Function that runs the fully dp topic model proceadure
    Inputs: 
        - corpus (list): list of documents (str)
        - dpsu_func (function): function thay does dp vocabulary selection
        - dpsu_func_params (dict): parameters for dpsu function
        - dp_train_func (function): function that does dp topic modeling
        - dp_train_params (dict): parameters for dp topic modeling algorithm
    Outputs:
        - results (dict): dictionary with results of test
    """
    # Choose the vocabulary set
    dp_vocab_set = choose_vocab(corpus = corpus, **dpsu_func_params)
    
    # Sanitize the corpus
    clean_corpus = sanitize_corpus(dp_vocab_set, corpus)
    
    # Get doc-term matrix
    X, idx2word = create_doc_term_mat(clean_corpus)
    
    #fit model
    phi, logll, perp = dp_train_func(X = X, **dp_train_params)
    
    #get topic coherence and likelihood estimates
    coherence_10 = []
    coherence_30 = []
    for i in range(dp_train_params['k']):
        coherence_10.append(get_topic_coherence(X = X, phi = phi, topic_idx=i, n=10))
        coherence_30.append(get_topic_coherence(X = X, phi = phi, topic_idx=i, n=30))
    
    results = {'phi': phi, 'idx2word': idx2word, 'vocab_length': len(idx2word),
               'dpsu_func': dpsu_func.__name__, 'dp_train_func': dp_train_func.__name__,
              'coherence10': coherence_10, 'logll': logll, 'perp': perp, 'coherence_30': coherence_30}
    results['dp_train_params'] = dp_train_params
    results['dpsu_func_params'] = dpsu_func_params
    return results

if __name__ == "__main__":
    
    data = [["dog cat"], ["cat dog"], ["moose dog"], ["tree"]]
    sim_args = {'corpus': data,
                'dpsu_func': choose_vocab,
                'dpsu_func_params': {'alpha_cutoff': 3, 'epsilon': 100, 'delta':10e-5},
                'dp_train_func': train_dp_gibbs,
                'dp_train_params': {'k':5, 'iterations':15, 'epsilon':100}
           }
    
    sim_out = run_fdptm_sim(**sim_args)