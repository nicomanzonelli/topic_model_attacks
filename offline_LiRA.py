"""
This file provides the base class and functions for the offline LiRA
"""
import numpy as np
from scipy.stats import norm
from .utils import get_doc_word_idx

class LiRA_offline:
    
    def __init__(self, attack_phi, shadow_phis, target_document):
        """
        Initialize the class with
            - attack_phi (array): 2-d array of topic word distribution
            - shadow_phis (list of array): list of trained shadow phis
            - target_document (array): 1-d array of document vector
        """
        self.attack_phi = attack_phi
        self.shadow_phis = shadow_phis
        self.target_document = target_document
        
    def _calculate_statistics(self, stat_func):
        """
        Function to compute the statistics of interest for given phis
        Inputs:
            - self
            - stat_func (function): function that returns statistic (float)
        Output: Creates class attributes
            - out_stats (array): Array of shadow phis statistics
            - out_stats (float): Observed (attack model) statistics
        """
        # Get target doc word indicies
        target_w_idx = get_doc_word_idx(self.target_document)
        
        # Compute Statistics
        out_stats = np.array([stat_func(phi, target_w_idx) for phi in self.shadow_phis])
        obs_stats = np.array(stat_func(self.attack_phi, target_w_idx))
        
        # Assign out to class attribute
        self.out_stats = out_stats
        self.obs_stats = obs_stats
    
    def _calculate_lambda(self):
        """
        Function to calculate lambda the test statisitc
        Inputs:
            -self
        Creates new class attribute:
            - lambda_offline (float): test stastic for the one-sided hypothesis test
        """
        # Estimate normal
        norm_without = self.estimate_normal(self.out_stats)
        
        # Calculate p-value
        self.lambda_offline = 1 - norm_without.sf(self.obs_stats)
        
    def fit(self, stat_func):
        """
        Function to fit the offline LiRA
        """
        self._calculate_statistics(stat_func)
        self._calculate_lambda()
        return self.lambda_offline
    
    @staticmethod
    def estimate_normal(array):
        """
        Function that estimates a normal distribution given an array of values.
        Inputs: 
            array (np.array): Array of values.
        Outputs:
            - norm (scipy instance of the rv_continuous class): A normal continuous random variable.
        """
        return norm(loc=np.mean(array), scale=np.std(array))
    
    