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
    return vocab