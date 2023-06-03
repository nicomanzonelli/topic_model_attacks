from .attack_simulations import simple_exp, synth_exp
from .utils import read_json, write_pickle, create_doc_term_mat

import json
import pickle

def run_simple_exp(in_path, out_path, sim_args):
    """
    
    """
    # Read in training data
    data = read_json(in_path)
    
    # Create doc-term count matrix
    raw_corpus = data['text']
    X, idx2word = create_doc_term_mat(raw_corpus)
    sim_args['X'] = X
    
    # Run simple sim
    sim_out = simple_exp(**sim_args)
    
    # Clean up output
    sim_args['data_path'] = in_path
    sim_args['X_shape'] = X.shape
    sim_args['idx2word'] = idx2word
    sim_args["attack_training_function"] = sim_args["attack_training_function"].__name__
    sim_args["shadow_training_function"] = sim_args["shadow_training_function"].__name__
    sim_args["stat_func"] = sim_args["stat_func"].__name__
    sim_out = {**sim_args, **sim_out}
    del sim_out['X']
    del sim_out['shadow_params']['X']
    del sim_out['attack_params']['X']
    
    # Write results
    write_pickle(sim_out, out_path)
    
if __name__ == "__main__":
    from utils import train_model_sklearn, calc_max_logll

    sim_args = {"train_sample_p": 0.5, 
        "N": 256, 
        "attack_training_function": train_model_sklearn, 
        "attack_params": {"k": 5},
        "shadow_training_function": train_model_sklearn, 
        "shadow_params": {"k": 5}, 
        "stat_func": calc_max_logll, 
        "verbose": True}

    in_path = './data/pheme_clean.json'
    out_path = f'./evals/tweet_128sm_{it}.pickle'
    sim_out = run_simple_exp(in_path, out_path, sim_args)
    
    
    