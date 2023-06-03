"""
This file provides the base class and functions for the online LiRA
"""
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from .utils import get_doc_word_idx

# To handle errors on ratio
import warnings
warnings.filterwarnings("error")

class LiRA_online:
    
    def __init__(self, attack_phi, in_shadow_phis, out_shadow_phis, target_document):
        """
        Initialize the class with
            - attack_phi (array): 2-d array of topic word distribution
            - in_shadow_phis (list of array): list of trained shadow phis
            - out_shadow_phis (list of array): list of trained shadow phis
            - target_document (array): 1-d array of document vector
        """
        self.attack_phi = attack_phi
        self.in_shadow_phis = in_shadow_phis
        self.out_shadow_phis = out_shadow_phis
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
        in_stats = np.array([stat_func(phi, target_w_idx) for phi in self.in_shadow_phis])
        out_stats = np.array([stat_func(phi, target_w_idx) for phi in self.out_shadow_phis])
        obs_stats = np.array(stat_func(self.attack_phi, target_w_idx))
        
        # Assign out to class attribute
        self.in_stats = in_stats
        self.out_stats = out_stats
        self.obs_stats = obs_stats
    
    def _calculate_lambda(self):
        """
        Inputs:
            - self
        Creates new class attribute:
            - lambda_online (float): Test statstic for LR test.
        """
        # Estimate Normals
        norm_in = self.estimate_normal(self.in_stats)
        norm_out = self.estimate_normal(self.out_stats)
        
        # Evaluate pdf at observed value
        norm_in_pdf =  self.fix_p(norm_in.pdf(self.obs_stats))
        norm_out_pdf = self.fix_p(norm_out.pdf(self.obs_stats))
        
        # Calculate ratio. Need to catch when ratio is inf
        try:
            lambda_online = float(norm_in_pdf / norm_out_pdf)
        except RuntimeWarning:
            lambda_online = float(sys.float_info[0])
        
        self.lambda_online = lambda_online
            
        
    def _calculate_lambda_offline(self):
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
        
    def fit(self, stat_func, offline = False):
        """
        Function to fit the online LiRA (with option to compute offlines)
        """
        self._calculate_statistics(stat_func)
        self._calculate_lambda()
        if offline:
            self._calculate_lambda_offline()
        return self.lambda_online
       
    def plot(self):
        norm_with = self.estimate_normal(self.in_stats)
        norm_without = self.estimate_normal(self.out_stats)
        max_std = 3*max(np.std(self.in_stats), np.std(self.out_stats))
        upper = max(max(self.in_stats), max(self.out_stats)) + 3*max_std
        lower = min(min(self.in_stats), min(self.out_stats)) - 3*max_std
        approx_range = np.linspace(lower, upper, 1000)
        plt.plot(approx_range, norm_with.pdf(approx_range), label = "Shadow $\mathcal{N}_{in}$")
        plt.plot(approx_range, norm_without.pdf(approx_range), label = "Shadow $\mathcal{N}_{out}$")
        plt.axvline(x = self.obs_stats, label = "Attack Model $f(\Phi_{obs})$", color = 'red')
        plt.title(f"Estimated Normals for Shadow Models")
        plt.show()
    
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
    
    @staticmethod
    def fix_p(p):
        """
        Function that helps process probabilty estimates. Technically, the pdf of
        a distribution is never zero. When scipy.pdf returns a 0 it's because the pdf
        is extreamly small and rounds automatically. We fix that.
        Input:
            - p (float): pdf
        Output
            - (float): minimum float representation for system
        """
        if p == 0:
            return sys.float_info[3] #minimum system float
        else: 
            return p
    
    @staticmethod
    def handle_inf(test_statistic):
        """
        Function that handles the test statistic returnning inf. This occurs when the
        denominator is a very small number (smallest float representation) and the 
        numerator is larger.
        """
        if test_statistic == np.inf:
            return sys.float_info[0] #Return the maximum float representation
        else:
            return test_statistic


    