"""A python module for loading text data and working with vocabulary sets.

"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def load_data_json(data_path):
    """
    Function that loads data from the json to a list of documents.

    Args:
        - data_path (str): Path to .json file with list of documents in the 'text'
            field.
    Returns:
        - corpus (list(str)): A list of documents represented as strings.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    return data['text']

def extract_features(corpus, vocab = None):
    """
    Function that loads data from the json to a document-term matrix and
    returns the vocabulary set in a tuple.

    Args:
        - corpus: A list of documents represented as strings.
        - vocab (np.ndarray[str]): An array of the vocabulary to represent docs.
            
    Returns:
        - X: The document-term matrix with shape (num docs x vocab length)
        - vocabulary: An array of terms (str) whose indices correspond to the
            columns of X.
    """
    if vocab is not None:
        vectorizer = CountVectorizer(vocabulary = vocab)
    else:
        vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(corpus).toarray()
    vocab = vectorizer.get_feature_names_out()
    return X, vocab

def encode_doc(doc_arr):
    """
    A function that takes a document represented as an array of size vocabulary
    length whose entries correspond to vocab frequency (basically just a row of 
    the document-term matrix) and returns the document encoded as a series of
    vocab indices.
    
    Args:
        - doc_arr (np.ndarray): Array of vocabulary length with doc term frequency

    Returns:
        - doc_encoded (np.ndarray): An array of vocabulary indices.
    """
    nonzero_indices = np.nonzero(doc_arr)[0]
    return np.repeat(nonzero_indices, doc_arr[nonzero_indices])

def split_data(data, p = .5):
    """
    A function that randomly selects indices to include in a split

    Args:
        - data (list): A collection of documents represented as a list of strings.
        - p: The proportion of the documents to include in the split [0,1].
            default to .5.

    Returns:
        - (np.ndarray) Array containing a boolean indicator to represent the documents
            inclusion in split.
    """
    n = len(data)
    split_size = int(np.ceil(n * p))
    in_split = np.hstack((np.ones(split_size), np.zeros(n - split_size)))
    np.random.shuffle(in_split)
    return in_split.astype(bool)


