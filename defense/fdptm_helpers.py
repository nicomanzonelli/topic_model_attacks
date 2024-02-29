""" fdptm_helpers.py

Some helper functions to assist in implementing Fully Differentially Private
Topic Modeling

"""

from .dpsu_gw import run_gw

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
    return list(vocab.keys())

def get_coherence(X, phi, topic_idx, n, smoothing_param = 1):
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
