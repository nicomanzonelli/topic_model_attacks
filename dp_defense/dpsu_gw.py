"""
Code for DP Set Union from https://github.com/ricardocarvalhods/diff-private-set-union

From Carvalho, R. S., Wang, K., & Gondara, L. S. (2022).
"""


import numpy as np
from tqdm import tqdm
import operator
from collections import Counter, defaultdict, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    pass

def run_gw(input_data, alpha, eps_ours, delta_ours, greedy, kt_data, verbose=False):
    ## Greedy
    if greedy not in ('ci', 'kt-ci'):
        raise Exception("ERROR: Greedy not allowed!")
    
    ## Cutoff and noise parameters
    l_param = 1 / eps_ours
    l_rho = 1 - (1 / eps_ours) * np.lib.scimath.log( 2 * delta_ours )
    Gamma = l_rho + alpha*l_param    
    
    ## Added shuffling
    #np.random.seed(1)
    np.random.shuffle(input_data)

    new_hist = defaultdict(float)

    for all_grams in tqdm(input_data, position=0, leave=True, disable=(not verbose)):

        ## Filter below cutoff
        selected_ngrams = [gram for gram in all_grams if new_hist[gram] < Gamma]

        if len(selected_ngrams) > 0:
            # Calculate weights to define order of items
            if greedy == 'ci':
                all_counter = OrderedCounter(all_grams)
            elif greedy == 'kt-ci':
                user_weight = np.array([kt_data[sel] if sel in kt_data else 1 for sel in selected_ngrams])

            rho_dict = {}
            for i, w in enumerate(selected_ngrams):
                if new_hist[w] < Gamma:
                    if greedy == 'ci':
                        rho_dict[w] = all_counter[w]
                    elif greedy == 'kt-ci':
                        rho_dict[w] = user_weight[i]

            # Start with argmax
            sorted_rho_dict = sorted(rho_dict.items(), key=operator.itemgetter(1), reverse=True)
            sorted_rho_keys = [k for k, v in sorted_rho_dict]

            # Update greedily
            user_budget = 1
            for i, w in enumerate(sorted_rho_keys):
                cost = Gamma - new_hist[w]

                if cost < user_budget:
                    new_hist[w] = new_hist[w] + cost
                    # update remaining user_budget
                    user_budget -= cost
                else:
                    # not enough user_budget to meet threshold: add rest of user_budget to item
                    new_hist[w] = new_hist[w] + user_budget
                    break

    ## Get items with noisy weights above threshold
    output_vocab = {}
    for key, val in new_hist.items():
        nval = val + np.random.laplace(0, l_param)
        if nval > l_rho:
            output_vocab[key] = val

    return output_vocab
