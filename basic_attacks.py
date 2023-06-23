"""
This file implements the basic attack against topic models in accordance
Huang et. al (2022), "Improving Parameter Estimation and Defensive Ability of 
Latent Dirichlet Allocation Model Training Under Rényi Differential Privacy."
"""
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

from utils import get_doc_word_idx, sample_documents, est_doc_topic_dist, calc_entropy

"""
Functions to carry out the attack against topic models
"""
def basic_attack(X, p_train, attack_training_function, attack_model_params):
    """
    A function to run a basic simulation where the steps are as follows:
        - Choose the attack model trainset from the dataset X with p_train prob.
        that any doc is in the trainset
        - Train an attack model on trainset and release the topic-word distribution
        - Randomly sample eval_sample_n documents in X as target_documents
        - Randomly sample aux_sample_n from X as as the auxilary set.
        - For each target document fit a LiRA using the the auxilary set as the
            adversaries data.
    Inputs:
        - X (array): Data used for attack model, target documents, and shadow models.
        - p_train (float): probability that any document of X is in the trainset
        - attack_training_function (function): Function that trains the attack
            model and outputs a topic-word distribution (phi)
        - attack_model_params (dict): Parameters for the attack training function
        - eval_sample_n (int): The number of documents sampled from X as target documents
        - aux_sample_n (int): The number of documents sampled from the auxilary dataset
            to be used for training shadow models.
        - max_ps (list): maximum posteriros of estimated doc topic dist for each target doc
        - stds (list): standard deviation of estimated doc topic dist for target each doc
            - entropys (list): entropy of estimated target doc topic dist for target each doc
    Output:
        - out (dict): A dictionary that contains the all of the inputs and...
         - target_doc_truth (list of bool): Indicate if the target doc was in train set.
         - eval_docs_idx (list): List of document indicies sampled for LiRA
         - max_ps (list): 
         - stds (list): 
         - entropys (list): 
         - target_docs_truth (list): 
         - attack_train_docs_idx (list): 
         - ... some inputs for reference
    """
    train_docs, train_docs_idx = sample_documents(X, p_train)

    # Train attack model
    print("Training Attack Model")
    # Reset or set X for training purposes
    attack_model_params['X'] = train_docs
    attack_phi = attack_training_function(**attack_model_params)

    # Choose target_documents
    target_docs_truth = 1 * np.in1d(np.arange(X.shape[0]), train_docs_idx)

    # Define lists for storing results
    max_ps = []
    stds = []
    entropys = []
    print("Calculating Metrics on Target Model")
    for target_doc_idx in tqdm(np.arange(X.shape[0])):
        # Define the target document
        target_doc = X[target_doc_idx, :]
        target_w_idx = get_doc_word_idx(target_doc)

        # Estimate Target Document's Topic Dist
        theta_hat = est_doc_topic_dist(attack_phi, target_w_idx)

        # Calculate Statistics
        max_ps.append(np.max(theta_hat))
        stds.append(np.std(theta_hat))
        entropys.append(entropy(theta_hat))

    # Including datasets makes a huge outfile
    attack_model_params['X'] = []
    out = {"p_train": p_train,
           "attack_training_function": str(attack_training_function.__name__),
           "max_ps": max_ps,
           "stds":stds,
           "entropys": entropys,
           "target_docs_truth": target_docs_truth.tolist(),
           "attack_train_docs_idx": train_docs_idx.tolist(),
          }
    return out

if __name__ == "__main__":
    ###TESTING
    from utils import read_json, write_pickle, create_doc_term_mat, train_model_sklearn

    # Read in dataset
    data = read_json('./data/pheme_clean.json')
    
    X, _ = create_doc_term_mat(data['text'])

    #kwargs for basic simulation
    kwargs = {"X": X, "p_train": .5,
              "attack_training_function": train_model_sklearn,
              "attack_model_params": {"k":5}
             }
    
    sim_out = basic_attack(**kwargs)
    path = f'./pheme_basic_test.pickle'
    sim_out['path'] = path
    write_pickle(sim_out, path)
        