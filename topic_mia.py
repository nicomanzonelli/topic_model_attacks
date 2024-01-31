""" topic_mia.py
This file provides the base class and functions for the Likelihood Ratio Attack
from Carlini et al. "Membership Inference Attacks from First Principles" applied
to topic models
"""

import os
import sys
import functools
import multiprocessing as mp
from typing import List, Callable, Tuple

import numpy as np
from scipy.stats import norm, multivariate_normal

from data_utils import extract_features, encode_doc

class TopicLiRA:
    """
    The base class for the LiRA on topic models.
    """
    def __init__(self, target_phi: np.ndarray, 
                 target_vocabulary: np.ndarray):
        """
        The TopicLiRA is initialized with the following:
            - target_phi (ndarray): The target model's topic-word distribution
                of shape (num topics x length of vocabulary)
            - target_vocabulary (ndarray): The target model's vocabulary set
                of shape (length of vocabulary,). Each entry corresponds to 
                a word indexed in the vocabulary.
            - offline_only (bool): Boolean indicator to run only the offline test.

        Other class attribute include:
            - self.in_stats (ndarray):
            - self.out_stats (ndarray):
            - self.obs_stats (ndarray):
            - self.scores (Tuple(list, list)):
        """
        self.target_phi = target_phi
        self.target_vocabulary = target_vocabulary
        self.in_stats = None
        self.out_stats = None
        self.obs_stats = None
        self.scores = None

    def fit(self, target_documents: List[str],
            shadow_phis: np.ndarray,
            docs_in_shadow: List[np.ndarray],
            statistic_functions: List[Callable],
            each_stat: bool = False,
            offline_only: bool = False,
            enable_mp: bool = True,
            out_path: str = None) -> np.ndarray:
        """
        A function that fits a LiRA to a set of documents or observations.
        
        Args:
            - target_documents (list(str)): A set of documents or obs to test for
                membership in the target model!

            - shadow_phis (ndarray): All of the shadow topic models trained
                in one (k x len(vocab) x num shadow models) shaped array.

            - docs_in_shadow (List(ndarray)): A list of of the target document
                indices included in each shadow model. The n^th list entry is an
                array of target_document indices included in the n^th shadow model.

            - statistic_functions (List(function())): A list of statistic functions. 
                These functions must take a topic-model (phi) and an encoded document
                (a document encoded as an array of the vocabulary indices of the words
                in the doc) and return a float (the model query statistic). This can
                also return a list of floats and it will be flattened in execution.

            - each_stat (bool): If false uses all stats to calculate one score 
                with mvnorm. If true it iterates through each statistic and produces 
                a separate score for each candidate statistic. The resulting array 
                is (num_statistics + 1 x num obs) shape. The entries are 
                the score for each document under each statistic. The +1 column 
                exists because we include the score using mvnorm as the first column 
                then subsequent columns are the scores for each statistic in order.

            - offline_only (bool): Boolean indicator to only run offline test.

            - enable_mp (bool): Enables multiprocessing to calculate statistics.
                Useful for running in jupyter for debugging or small tests.

            - out_path (str): A location to save artifacts or checkpoints from
                the attack. File saved will be:
                out_path/
                └ in_stats.npy - array shaped (len(obs) x ? x len(stat_funcs))
                |   contains the query statistics for each document for the shadow
                |   models that the document is included in.
                └ out_stats.npy - same as in_stats but for models the document is
                |   not in.
                └ obs_stats.npy - array shaped (len(obs) x len(stat_funcs))
                |   containing the query statistics for each document.
                └ offline_scores.npy - array shaped (len(obs),) with offline MIA scores
                └ online_scores.npy - array shaped (len(obs),) with online MIA scores
        """
        if out_path:
            os.makedirs(out_path, exist_ok=True)

        X, _ = extract_features(target_documents, self.target_vocabulary)

        # Get all statistics
        self._get_statistics(X, shadow_phis, docs_in_shadow, statistic_functions, enable_mp)

        if out_path:
            np.save(os.path.join(out_path, 'in_stats.npy'), self.in_stats)
            np.save(os.path.join(out_path, 'out_stats.npy'), self.out_stats)
            np.save(os.path.join(out_path, 'obs_stats.npy'), self.obs_stats)

        offline_scores = self._score_offline(each_stat)
        scores = (offline_scores, None)

        if out_path:
            np.save(os.path.join(out_path, 'offline_scores.npy'), offline_scores)

        if not offline_only:
            online_scores =  self._score_online(each_stat)

            if out_path:
                np.save(os.path.join(out_path, 'online_scores.npy'), online_scores)

            scores = (offline_scores, online_scores)
        
        self.scores = scores
        return scores

    def _get_statistics(self, X: np.ndarray, 
                        shadow_phis: np.ndarray, 
                        docs_in_shadow: List[np.ndarray], 
                        stat_funcs: List,
                        enable_mp: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function that gets the statistics for each target document and each shadow
        model.
        """

        map_kwargs = {'shadow_phis': shadow_phis, 'docs_in_shadow': docs_in_shadow,
                  'stat_funcs': stat_funcs, 'target_phi': self.target_phi}
        map_partial = functools.partial(map_calculate_stats, **map_kwargs)
    
        args_list = [(i, encode_doc(X[i])) for i in range(X.shape[0])]

        if enable_mp:
            with mp.Pool(mp.cpu_count()) as pool:
                map_return = pool.map(map_partial, args_list)

        else:
            map_return = [map_partial(arg) for arg in args_list]

        self.in_stats = np.array([map_return[i][0] for i in range(X.shape[0])])
        self.out_stats = np.array([map_return[i][1] for i in range(X.shape[0])])
        self.obs_stats = np.array([map_return[i][2] for i in range(X.shape[0])])

    def _score_online(self, each_stat):
        """
        
        """
        scores = self._online_mvnorm_score()

        if each_stat:
            scores = [scores] + self._online_score_each()

        return scores

    def _online_mvnorm_score(self):
        """
        
        """
        scores = []

        for i in range(self.obs_stats.shape[0]):
            in_mean = self.in_stats[i,:,:].mean(axis = 0)
            in_var = self.in_stats[i,:,:].var(axis = 0)+10e-32 # Keep from zero
            out_mean = self.out_stats[i,:,:].mean(axis = 0)
            out_var = self.out_stats[i,:,:].var(axis = 0)+10e-32

            in_mvnorm = multivariate_normal(in_mean, in_var, allow_singular=True)
            out_mvnorm = multivariate_normal(out_mean, out_var, allow_singular=True)
            in_pr = in_mvnorm.logpdf(self.obs_stats[i])
            out_pr = out_mvnorm.logpdf(self.obs_stats[i])
            scores.append(np.exp(in_pr - out_pr))

        return scores

    def _online_score_each(self):
        """

        """
        scores = []
        for si in range(self.obs_stats.shape[2]):
            stat_scores = []
            for di in range(self.obs_stats.shape[0]):
                in_mean = self.in_stats[di,:,si].mean(axis = 0)
                in_std = self.in_stats[di,:,si].std(axis = 0)+10e-32
                out_mean = self.out_stats[di,:,si].mean(axis = 0)
                out_std = self.out_stats[di,:,si].std(axis = 0)+10e-32
                
                in_norm = norm(in_mean, in_std)
                out_norm = norm(out_mean, out_std)
                in_pr = in_norm.logpdf(self.obs_stats[di, si])
                out_pr = out_norm.logpdf(self.obs_stats[di, si])
                stat_scores.append(np.exp(in_pr - out_pr))
        
            scores.append(stat_scores)

        return scores

    def _score_offline(self, each_stat):
        """
        
        """
        scores = self._offline_mvnorm_score()

        if each_stat:
            scores = [scores] + self._offline_score_each()

        return scores
    
    def _offline_mvnorm_score(self):
        """
        
        """
        scores = []

        for i in range(self.obs_stats.shape[0]):
            out_mean = self.out_stats[i,:,:].mean(axis = 0)
            out_var = self.out_stats[i,:,:].var(axis = 0)+10e-32

            out_mvnorm = multivariate_normal(out_mean, out_var, allow_singular=True)
            scores.append(out_mvnorm.cdf(self.obs_stats[i]))
        
        return scores

    def _offline_score_each(self):
        """
        
        """
        scores = []
        for si in range(self.obs_stats.shape[2]):
            stat_scores = []
            for di in range(self.out_stats.shape[0]):
                out_mean = self.out_stats[di,:,si].mean(axis = 0)
                out_std = self.out_stats[di,:,si].std(axis = 0)+10e-32
                
                out_norm = norm(out_mean, out_std)
                stat_scores.append(out_norm.cdf(self.obs_stats[di, si]))

            scores.append(stat_scores)

        return scores
    
# Helper Functions

def map_calculate_stats(args, shadow_phis, target_phi, docs_in_shadow, stat_funcs):
    """Helper function to calculate stats for one document. Called in pool.map
    in _get_statistics. Need separate for serialization?
    """
    idx, doc = args
    ins = []
    outs = []
    for m_i in range(shadow_phis.shape[2]):
        if idx in docs_in_shadow[m_i]:
            ins.append(fltn([f(shadow_phis[:,:,m_i], doc) for f in stat_funcs]))
        else:
            outs.append(fltn([f(shadow_phis[:,:,m_i], doc) for f in stat_funcs]))

    obs = fltn([f(target_phi, doc) for f in stat_funcs])

    return ins, outs, obs   

def fltn(lst):
    """Helper function which flattens a list (at most-one level deep)"""
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out 
