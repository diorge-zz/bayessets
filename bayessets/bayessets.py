"""
.. module:: bayessets
    :synopsis: Implementations of the Bayesian Sets algorithms
"""


import numpy as np
from scipy.sparse import csr_matrix


class BernoulliBayesianSet:
    """Bayesian Sets assuming an independent Bernoulli distribution.
    Using the conjugate distribution (Beta),
    we build an efficient computation model
    based on a matrix multiplication.

    The model needs hyperparameters alpha and beta,
    but these can be estimated using the mean of each feature.
    To use this estimation, the meanfactor argument should
    be set, while alpha and beta should be None.

    The alphaepsilon and betaepsilon arguments
    are used for features with mean (close to) 0 or 1,
    because they will cause log-of-zero operations.
    One option in this case is to look into these features,
    but if even then the floating-point precision,
    the epsilon arguments could be used to correct the issue.
    """
    def __init__(self, dataset, meanfactor=2, alpha=None, beta=None,
                 alphaepsilon=0, betaepsilon=0):
        """
        :param dataset: binary matrix, with samples on the horizontal
                        and features on the vertical
        :type dataset: matrix-like (will be converted to Scipy's CSR matrix)
        :param meanfactor: scalar to be multiplied by means to set alpha
        :type meanfactor: float
        :param alpha: alpha parameter vector for the Beta distribution
        :type alpha: array-like
        :param beta: beta parameter vector for the Beta distribution
        :type beta: array-like
        :param alphaepsilon: small value to be added to alpha to prevet log(0)
        :type alphaepsilon: float or array-like
        :param betaepsilon: small value to be added to beta to prevent log(0)
        :type betaepsilon: float or array-like
        """
        self.dataset = csr_matrix(dataset)
        if alpha is None and beta is None:
            self.alpha, self.beta = estimate_beta_parameters(dataset,
                                                             meanfactor)
        else:
            self.alpha = alpha
            self.beta = beta
        self.alpha += alphaepsilon
        self.beta += betaepsilon
        self.alpha_plus_beta = self.alpha + self.beta
        self.log_alpha = np.log(self.alpha)
        self.log_beta = np.log(self.beta)
        self.log_alpha_plus_beta = np.log(self.alpha_plus_beta)

    def query(self, query_indices):
        """Computes the expansion of the seed set given by
        the argument query_indices

        :param query_indices: list of the indices of items in the seed set
        :returns: ndarray -- the score of each item
        """
        rankconstant, rankquery = self.compute_query_parameters(query_indices)
        log_scores = rankconstant + self.dataset * rankquery.transpose()
        return np.asarray(log_scores.flatten())[0]

    def query_many(self, queries):
        """Computes the expansion of the seed sets given
        in the argument queries.
        Multiplies the dataset matrix only once,
        may be recommended for larger/more dense matrices.

        :param queries: list of the query indices
        :returns: matrix -- the score of each item for each query
        """
        rankconstants = []
        rankqueries = []
        for query in queries:
            rankconstant, rankquery = self.compute_query_parameters(query)
            rankconstants.append(rankconstant)
            rankqueries.append(rankquery)
        rankconstants = np.array(rankconstants)
        rankqueries = np.vstack(rankqueries)
        score_matrix = rankconstants + self.dataset * rankqueries.T
        return score_matrix.T

    def compute_query_parameters(self, query_indices):
        """Computes the query parameters, rank constant and rank query
        (called 'c' and 'q' in the original paper, respectively)
        """
        querysize = len(query_indices)
        sum_x = np.sum(self.dataset[query_indices], axis=0)
        alpha_tilde = sum_x + self.alpha
        beta_tilde = self.beta + querysize - sum_x
        log_alpha_tilde = np.log(alpha_tilde)
        log_beta_tilde = np.log(beta_tilde)
        rankconstant = (self.alpha_plus_beta -
                        np.log(self.alpha + self.beta + querysize) +
                        log_beta_tilde - self.log_beta
                        ).sum()
        rankquery = (log_alpha_tilde - self.log_alpha -
                     log_beta_tilde + self.log_beta)
        return rankconstant, rankquery


def estimate_beta_parameters(dataset, meanfactor=2):
    """Estimates alpha and beta parameters for Beta distribution
    by using the mean of each feature
    """
    alpha = meanfactor * dataset.mean(0)
    beta = meanfactor - alpha
    return alpha, beta
