import numpy as np
from scipy.special import gammaln

class LDA:
    """
    Base Class to fit LDA using Collapsed Gibbs Sampling derived from 
    Griffiths and Steyvers (2004): Finding scientific topics.
    Based on https://github.com/chriswi93/LDA/blob/main/lda/base.py
    """
    
    def __init__(self, doc_term_matrix, k, alpha, beta):
        """
        Initialize LDA class.
        Inputs:
            doc_term_matrix (np.array): Matrix with dims (# of docs x # of words in vocab). Counts words in vocab for each document.
            k (int): number of topics
            alpha (float): dirichlet prior for theta
            beta (prior): dirichlet prior for phi
        """
        
        #init variables
        self.k = k
        self.dtm = doc_term_matrix
        self.W = doc_term_matrix.shape[1] # number of words in vocabulary
        self.M = doc_term_matrix.shape[0] # number of docs in corpus
        self.dwc = self.dtm.sum(axis = 1) # count number of words in each document
        
        self.alpha = alpha
        self.beta = beta
        
        #init topic-word count matrix (k x V)
        self.wtcm = np.zeros((k, self.W))
        
        #init doc-topic count matrix (M x k)
        self.dtcm = np.zeros((self.M, self.k))
        
        # count vector of words assigned to each topic
        self.wtcv = np.zeros(self.k)
        
    def simple_fit(self, iterations):
        """
        Function that fits model for the desired number of iterations.
        Inputs: 
            - iterations (int): number of training iterations
        Outputs: 
            - phi (array): topic-word distribution
        """
        #initialization
        self._random_init()

        #sampling for each iteration
        for _ in range(iterations):
            self._sample()

        #compute phi and theta
        self._compute_theta()
        self._compute_phi()
        return self.phi
        
    def wl_private_fit(self, iterations, epsilon_l, C):
        """
        Function that fits a word level private LDA model.
        Presented in Zhoa et al (2021).
        Inputs:
            - iterations (int): the number of desired training iterations
            - epsilon_l (float): privacy loss parameter for noise added
            - C (float): Clipping bound for noise added to word-count
        Outputs:
            - phi (array): word-level private topic-word distribution
            - epsilon_g (float): global epsilon (total privacy loss)
        """
        #initialization
        self._random_init()

        #sampling for each iteration
        for _ in range(iterations):
            self._wl_private_sample(epsilon_l, C)

        #compute phi and theta
        self._compute_theta()
        self._compute_phi()
        
        #compute global privacy budget
        epsilon_g = iterations * (epsilon_l + 2*np.log((C/self.beta) +1))
        return self.phi, epsilon_g
    
    def dl_private_fit(self, iterations, epsilon):
        """
        Function that fits a document level private LDA model.
        Presented in Zhu et al (2016).
        Inputs:
            - iterations (int): number of gibbs sampling iterations.
            - epsilons (privacy parameter): global privacy parameter
        Outputs:
            - phi (array): private topic-word distribution
        """
        # Initialization
        self._random_init()

        # Sampling for each iteration
        for _ in range(iterations):
            self._sample()
            
        # Add noise to conditonal dist and resample!
        self._dl_private_sample(epsilon)
        
        #compute phi and theta
        self._compute_theta()
        self._compute_phi()
        
    def _random_init(self):
        """
        This function randomly init the topic assignments for each word in each document.
        This function updates all wtm, dtm, and wtv.
        Inputs: self
        Outputs: None. Creates new class attribute z.
            - self.z (list): nested list of tuples each inner list corresponds to a document
                            each inner list is composed of a tuple where each item is a tuple
                            (word_idx, topic_assignment).
        """
        z = []
        
        # for each document
        for doc_idx in range(self.dtm.shape[0]):
            doc = []
            #for each occurance of each word in document
            for word_idx in self.dtm[doc_idx].nonzero()[0]:
                for _ in range(self.dtm[doc_idx, word_idx]):
                    
                    #randomly sample a topic
                    topic_assignment = np.random.randint(self.k)
                    doc.append((word_idx, topic_assignment))
                    
                    #update counts
                    self.wtcv[topic_assignment] += 1
                    self.wtcm[topic_assignment, word_idx] += 1
                    self.dtcm[doc_idx, topic_assignment] += 1
                    
            z.append(doc)
        self.z = z
    
    def _sample(self):
        """
        Completes one full sampling step for LDA.
        Inputs: self
        Outputs: 
            - None. Updates count matricies and z.
        """
        for doc_idx in range(len(self.z)):
            #for each word in document
            for word_pos, (word_idx, topic_assignment) in enumerate(self.z[doc_idx]):

                #decrement counts without this word
                self.wtcv[topic_assignment] -= 1
                self.wtcm[topic_assignment, word_idx] -= 1
                self.dtcm[doc_idx, topic_assignment] -= 1

                #compute full conditional posterior
                p = self._compute_full_conditional(doc_idx, word_idx, topic_assignment)

                #sample a new topic from p
                new_topic_assignment = np.random.multinomial(1, p).argmax()

                #reassign word to new topic
                self.z[doc_idx][word_pos] = (word_idx, new_topic_assignment)

                #increment counts with word and new topic
                self.wtcv[new_topic_assignment] += 1
                self.wtcm[new_topic_assignment, word_idx] += 1
                self.dtcm[doc_idx, new_topic_assignment] += 1
    
    def _wl_private_sample(self, epsilon_l, C):
        """
        Completes one sampling step for word-level private LDA.
        Inputs:
            - epsilon_l (float): privacy loss parameter for noise added
            - C (float): Clipping bound for noise added to word-count
        Outputs:
            - None
        """
        for doc_idx in range(len(self.z)):
            #for each word in document
            for word_pos, (word_idx, topic_assignment) in enumerate(self.z[doc_idx]):
                
                #self.wtcm[topic_assignment, word_idx] is the count we must add noise to
                # Note that we update this in place
                lap_noise = np.random.laplace(loc = 0.0, scale = (2/epsilon_l))
                self.wtcm[topic_assignment, word_idx] = self.wtcm[topic_assignment, word_idx] + lap_noise
                
                #decrement counts without this word
                self.wtcv[topic_assignment] -= 1
                self.wtcm[topic_assignment, word_idx] -= 1
                self.dtcm[doc_idx, topic_assignment] -= 1

                #compute full conditional posterior with our clipping bound
                p = self._compute_wlp_full_conditional(doc_idx, word_idx, topic_assignment, C)

                #sample a new topic from p
                new_topic_assignment = np.random.multinomial(1, p).argmax()

                #reassign word to new topic
                self.z[doc_idx][word_pos] = (word_idx, new_topic_assignment)

                #increment counts with word and new topic
                self.wtcv[new_topic_assignment] += 1
                self.wtcm[new_topic_assignment, word_idx] += 1
                self.dtcm[doc_idx, new_topic_assignment] += 1
                
    def _dl_private_sample(self, epsilon):
        """
        Completes one sampling step for doc-level private LDA.
        This should only happend one time at the very end of the sampling period
        Inputs:
            - epsilon_l (float): privacy loss parameter for noise added
            - C (float): Clipping bound for noise added to word-count
        Outputs:
            - None
        """
        for doc_idx in range(len(self.z)):
            #for each word in document
            for word_pos, (word_idx, topic_assignment) in enumerate(self.z[doc_idx]):

                #decrement counts without this word
                self.wtcv[topic_assignment] -= 1
                self.wtcm[topic_assignment, word_idx] -= 1
                self.dtcm[doc_idx, topic_assignment] -= 1

                #compute full conditional posterior
                p = self._compute_dlp_full_conditional(doc_idx, word_idx, topic_assignment, epsilon)

                #sample a new topic from p
                new_topic_assignment = np.random.multinomial(1, p).argmax()

                #reassign word to new topic
                self.z[doc_idx][word_pos] = (word_idx, new_topic_assignment)

                #increment counts with word and new topic
                self.wtcv[new_topic_assignment] += 1
                self.wtcm[new_topic_assignment, word_idx] += 1
                self.dtcm[doc_idx, new_topic_assignment] += 1
        
    def _compute_full_conditional(self, doc_idx, word_idx, topic_assignment):
        """
        Function to compute the full conditional.
        Inputs: self
            - doc_idx (int): document index for current document
            - word_idx (int): word index of current word
            - topic_assignment (int): topic assignment of current word
        Outputs: 
            - p (array): a k dimensional vector where each entry represents
            probablity of sampling topic k for the word_idx
        """
        #get count of word_idx in each topic (k-dim vector)
        n_w_z = self.wtcm[:, word_idx]
        #get count of words in each topic
        n_z = self.wtcv
        #get counts of topic in doc_idx (k-dim vector)
        n_m_z = self.dtcm[doc_idx]
        #get count of words in doc_idx (int)
        n_m = self.dwc[doc_idx]
        #compute word-topic ratio (first part of full cond) (k-dim vector)
        wt_ratio = (n_w_z + self.beta) / (n_z + self.W * self.beta)
        #compute document-topic ratio (second part of full cond) (k-dim vector)
        td_ratio = (n_m_z + self.alpha) / (n_m + self.k * self.alpha)
        #compute full conditional
        p = wt_ratio * td_ratio
        #normalize
        p_norm = p / p.sum()
        return p_norm
    
    def _compute_wlp_full_conditional(self, doc_idx, word_idx, topic_assignment, C):
        """
        Function to compute the full conditional in the word-level private scenario.
        Inputs: self
            - doc_idx (int): document index for current document
            - word_idx (int): word index for current word
            - topic_assignment (int): topic assignment of current word
            - C (float): clipping bound
        Outputs: 
            - p (array): a k dimensional vector where each entry represents
            probablity of sampling topic k for the word_idx
        """
        #get count of word_idx in each topic (k-dim vector)
        n_w_z = self.wtcm[:, word_idx]
        #Clip this value to our clipping bound
        # Note this does not change the actual count. Refrenced as n_t_k temp in paper.
        n_w_z[topic_assignment] = min(n_w_z[topic_assignment], C)
        
        #then we continue the process as typical
        #get count of words in each topic
        n_z = self.wtcv
        #get counts of topic in doc_idx (k-dim vector)
        n_m_z = self.dtcm[doc_idx]
        #get count of words in doc_idx (int)
        n_m = self.dwc[doc_idx]
        #compute word-topic ratio (first part of full cond) (k-dim vector)
        wt_ratio = (n_w_z + self.beta) / (n_z + self.W * self.beta)
        #compute document-topic ratio (second part of full cond) (k-dim vector)
        td_ratio = (n_m_z + self.alpha) / (n_m + self.k * self.alpha)
        #compute full conditional
        p = wt_ratio * td_ratio
        #normalize
        p_norm = p / p.sum()
        return p_norm

    def _compute_dlp_full_conditional(self, doc_idx, word_idx, topic_assignment, epsilon):
        """
        Function to compute the full conditional in the document-level private scenario.
        Inputs: self
            - doc_idx (int): document index for current document
            - word_idx (int): word index for current word
            - topic_assignment (int): topic assignment of current word
            - epsilon (float)
        Outputs: 
            - p (array): a k dimensional vector where each entry represents
            probablity of sampling topic k for the word_idx
        """
        #Sample noise 
        eta1 = np.random.laplace(loc=0, scale = 2/epsilon)
        eta2 = np.random.laplace(loc=0, scale = 2/epsilon)
        
        #get count of word_idx in each topic (k-dim vector)
        n_w_z = self.wtcm[:, word_idx]
        #get count of words in each topic
        n_z = self.wtcv
        #get counts of topic in doc_idx (k-dim vector)
        n_m_z = self.dtcm[doc_idx]
        #get count of words in doc_idx (int)
        n_m = self.dwc[doc_idx]
        #compute word-topic ratio (first part of full cond) (k-dim vector)
        wt_ratio = (n_w_z + eta1 + self.beta) / ((n_z-eta1) + self.W * self.beta)
        #compute document-topic ratio (second part of full cond) (k-dim vector)
        td_ratio = (n_m_z + eta2 + self.alpha) / ((n_m-eta2) + self.k * self.alpha)
        #compute full conditional
        arr = wt_ratio * td_ratio
        #normalize
        p_norm = (arr-np.min(arr))/(np.max(arr)-np.min(arr)) / ((arr-np.min(arr))/(np.max(arr)-np.min(arr))).sum()
        return p_norm
        
                    
    def _compute_theta(self):
        """
        Function to compute theta our document-topic distribution based on topic assignments for each word.
        Inputs: self
        Output: None. Assigns new class attribute theta.
            - theta (array): a (M x k) matrix where each entry corresponds to the mixture of document in topic k.
        """
        theta = np.zeros((self.M, self.k))
        for d in range(self.M):
            for z in range(self.k):
                theta[d,z] = (self.dtcm[d,z] + self.alpha) / (self.dwc[d] + self.alpha*self.k)
        self.theta = theta
        
    def _compute_phi(self):
        """
        Function to compute phi our topic-word distribution based on topic assignments for each word.
        Inputs: self
        Output: None. Assigns new class attribute phi.
            - phi (array): a (k x V) matrix where each entry corresponds to the prob. word in topic K.
        """
        phi = np.zeros((self.k, self.W))
        for z in range(self.k):
            for w in range(self.W):
                phi[z,w] = (self.wtcm[z,w] + self.beta) / (self.wtcv[z] + self.beta*self.W)
        self.phi = phi
        
    def _compute_log_liklihood_perplexity(self):
        """
        Function that computes the joint log likelihood p(w, z) = p(w|z)p(z).
        Griffiths and Steyvers (2004): Finding scientific topics
        """
        log_likelihood = 0.0
        for z in range(self.k): # log p(w|z)
            log_likelihood += gammaln(self.W * self.beta)
            log_likelihood -= self.W * gammaln(self.beta)
            log_likelihood += np.sum(gammaln(self.wtcm[z,:] + self.beta))
            log_likelihood -= gammaln(np.sum(self.wtcm[z,:] + self.beta))
            
        for doc_idx in range(len(self.z)): # log p(z)
            log_likelihood += gammaln(self.k *self.alpha)
            log_likelihood -= self.k * gammaln(self.alpha)
            log_likelihood += np.sum(gammaln(self.dtcm[doc_idx,:] + self.alpha))
            log_likelihood -= gammaln(np.sum(self.dtcm[doc_idx,:] + self.alpha))
            
        perplexity = np.exp(- log_likelihood / self.dwc.sum(axis=0))
            
        return log_likelihood, perplexity
    
    def get_doc_topic_dist(self, document):
        """
        Function that retreives a documents document-topic mixture on any given document
        Inputs:
            - document: document represented as a vector (1, vocab_size)
        Outputs:
            - doc_topic_distribution: a vector representing the topic mixture for the topic (1, k)
        """
        pass
    
    def _compute_unnormalized_phi(self):
        """
        Function to compute phi our topic-word distribution based on topic assignments for each word.
        Inputs: self
        Output: None. Assigns new class attribute phi.
            - phi (array): a (k x V) matrix where each entry corresponds to the prob. word in topic K.
        """
        phi = np.zeros((self.k, self.W))
        for z in range(self.k):
            for w in range(self.W):
                phi[z,w] = (self.wtcm[z,w] + self.beta) # without normalization constant!
        return phi
    
        
        
        
        
        
                    
        
        
        
        

    