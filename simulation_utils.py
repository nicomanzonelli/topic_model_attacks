""" simulation_utils.py

Python module that helps run attack simulations

"""

# import python base modules
import json
import multiprocessing as mp
import os
import time
import functools

# import other req modules
import numpy as np

# import custom modules
from data_utils import load_data_json, split_data, extract_features
from topic_mia import TopicLiRA

def train_target_model(data_path, target_model_path, p, train_func, train_kwargs):
    """
    A function that trains and saves the target model and meta data associated
    with the target model.

    This function does not return anything but it does create the following files
    at `target_model_path`:
    - docs_in_target.npy (np.ndarry[bool]): Array with boolean indicator for 
        document at index is in target model (truth-vector).
    - vocabulary.npy (np.ndarray[object]): Array where each entry corresponds to 
        a word indexed in the vocabulary.
    - phi.npy (np.ndarray[float64]) A topic-word distribution with shape 
        (number of topics x len(vocabulary))
    - target_params.json: parameters and other metadata for learning model.

    Args:
        - data_path (str): Path to the data saved as a json.
            note: The way we structure attack artifacts, we require an even number
            of observations in each dataset. Its odd, but makes things faster.
        - target_model_path (str): Path to save results to.
        - p (float): Percentage of the data to include in the target model.
        - train_func (function()): A function that returns the topic-word dist.
            for all documents in selected for the target model.
        - train_kwargs (Dict): The key word arguments for the train function.

    Returns:
        - None
    """
    os.makedirs(target_model_path, exist_ok=True)

    params = {'p': p, 
              'data_path': data_path, 
              'train_func': train_func.__name__,
              'train_kwargs': train_kwargs}
    
    with open(os.path.join(target_model_path, 'target_params.json'), 'w') as f:
        json.dump(params, f)
    
    corpus = load_data_json(data_path)

    docs_in_target = split_data(corpus, p)
    np.save(os.path.join(target_model_path, 'docs_in_target'), docs_in_target)

    train_docs = np.array(corpus)[docs_in_target].tolist()

    X_tr, vocab = extract_features(train_docs)
    np.save(os.path.join(target_model_path, 'target_vocabulary'), vocab)

    train_kwargs['X'] = X_tr
    start_time = time.time()
    target_phi = train_func(**train_kwargs)
    total_time = time.time() - start_time
    print(f"[i] Trained target model in {total_time/60:.2f} minutes")
    np.save(os.path.join(target_model_path, 'target_phi'), target_phi)
    print(f"[i] Target model files saved to {target_model_path}")

def train_shadow_models(data_path, target_model_path, shadow_model_path, 
                        N, train_func, train_kwargs, enable_mp = True):
    """
    A function that splits the data in half N times and trains a shadow topic
    model based on each split of the data. We expect each document to appear 
    in N//2 shadow models.
    
    This function does not return anything but it does create the following files
    shadow_model_path.
    - shadow_phis.npy (np.ndarray[float]): Array of shape (k x len(vocab) x N) 
        it is a stack of each shadow model's topic-word distributions.
    - docs_in_shadow.npy (np.ndarray[int]): Array with shape (N x N//2).
        Each row corresponds to the document indecies included in the Nth shadow
        model.
    - shadow_params.json: parameters and other metadata for learning models.

    Args:
        - data_path (str): Path to the data saved as a json.
        - target_model_path (str): Path to the target model and its files.
        - shadow_model_path (str): Path to save shadow model files to.
        - N (int): The number of shadow models to train. Data is split N//2 times.
        - train_func (function()): A function that returns the topic-word dist.
            for all documents in selected for the shadow model.
        - train_kwargs (Dict): The key word arguments for the train function.
        - enable_mp (Bool): Boolean to enable multiprocessing.

    Returns:
        - None
    """
    os.makedirs(shadow_model_path, exist_ok=True)

    params = {'N': N, 
            'data_path': data_path, 
            'train_func': train_func.__name__,
            'train_kwargs': train_kwargs}
    
    with open(os.path.join(shadow_model_path, 'shadow_params.json'), 'w') as f:
        json.dump(params, f)

    vocab_path = os.path.join(target_model_path, 'target_vocabulary.npy')
    target_vocab = np.load(vocab_path, allow_pickle = True)

    corpus = load_data_json(data_path)
    X, _ = extract_features(corpus, target_vocab)

    # This loops through and splits the documents up evenly amongst each shadow
    # model
    docs_in_phi = []
    for _ in range(N//2):
        docs_in = np.random.choice(X.shape[0], X.shape[0]//2, replace=False)
        docs_out = np.setdiff1d(np.arange(X.shape[0]), docs_in)
        docs_in_phi.append(docs_in)
        docs_in_phi.append(docs_out)

    # We save docs_in_phi as an array (why the dataset must have even # of obs)
    docs_in_phi = np.array(docs_in_phi)
    np.save(os.path.join(shadow_model_path, 'docs_in_shadow'), docs_in_phi)

    map_kwargs = {'train_func': train_func, 'train_kwargs': train_kwargs}
    map_partial = functools.partial(map_train_shadow_model, **map_kwargs)
    
    start_time = time.time()
    # Shadow models run can train independently on CPU
    if enable_mp:
        pool = mp.Pool(mp.cpu_count()) 
        args_list = [X[docs_in_phi[i]] for i in range(N)]
        shadow_phis_list = pool.map(map_partial, args_list)
        pool.close()
        pool.join()
    else:
        shadow_phis_list = [map_partial(X[docs_in_phi[i]]) for i in range(N)]
    total_time = time.time() - start_time
    
    shadow_phis_path = os.path.join(shadow_model_path, 'shadow_phis.npy')
    print(f"[i] Trained {N} Shadow Models in {total_time/60:.2f} minutes")
    print(f"[i] Results saved to {shadow_model_path}.")
    
    shadow_phis = np.array(shadow_phis_list).transpose(1, 2, 0)
    np.save(shadow_phis_path, shadow_phis)

def map_train_shadow_model(args, train_func, train_kwargs):
    """ Helper function that trains exactly one shadow model. Called in pool.map
    in train shadow_models.
    """
    X_tr = args
    train_kwargs["X"] = X_tr
    shadow_phi = train_func(**train_kwargs)
    return shadow_phi

def fit_lira(data_path, target_model_path, shadow_model_path, fit_kwargs):
    """
    A function that fits a LiRA using the TopicLiRA defined in topic_mia.
    """
    # Load Data
    text_data = load_data_json(data_path)

    # Load Target
    target_phi = np.load(os.path.join(target_model_path, 'target_phi.npy'))
    vocab_path = os.path.join(target_model_path, 'target_vocabulary.npy')
    target_vocab = np.load(vocab_path, allow_pickle=True)

    # Init Attack
    mia = TopicLiRA(target_phi, target_vocab)

    # Load Shadow Models
    shadow_phis = np.load(os.path.join(shadow_model_path, 'shadow_phis.npy'))
    docs_in_shadow = np.load(os.path.join(shadow_model_path, 'docs_in_shadow.npy'))

    fit_kwargs['docs_in_shadow'] = docs_in_shadow
    fit_kwargs['shadow_phis'] = shadow_phis
    fit_kwargs['target_documents'] = text_data

    # This automatically saves the results to the out_path in fit_kwargs dict
    start_time = time.time()
    mia.fit(**fit_kwargs)
    total_time = time.time() - start_time

    print(f"[i] Fit LiRA in {total_time/60:.2f} minutes")
    print(f"[i] Results saved to {fit_kwargs['out_path']}")

if __name__ == "__main__":
    # This is some experimental code. The cli will run a simple experiment
    from topic_models import train_lda_sklearn
    from model_stats import max_dmm_logll

    stat_funcs = [max_dmm_logll]
    
    train_target_model("./data/pheme_clean.json",
                    f"./new_test/",
                    .5,
                    train_lda_sklearn,
                    {"k": 5})
    print()
    train_shadow_models("./data/pheme_clean.json",
                    f"./new_test/",
                    f"./new_test/",
                    64,
                    train_lda_sklearn,
                    {'k': 5})
    print()
    fit_kwargs = {'out_path': "./new_test/", 'statistic_functions': stat_funcs}
    fit_lira("./data/pheme_clean.json", "./new_test/", "./new_test/", fit_kwargs)


    

