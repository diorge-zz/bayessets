import numpy as np
from scipy.sparse import csr_matrix


class BernoulliBayesianSet:
    def __init__(self, dataset, alpha, beta):
        self.dataset = csr_matrix(dataset)
        self.alpha = alpha
        self.beta = beta
        self.alpha_plus_beta = self.alpha + self.beta
        self.log_alpha = np.log(self.alpha)
        self.log_beta = np.log(self.beta)
        self.log_alpha_plus_beta = np.log(self.alpha_plus_beta)

    def query(self, query_indices):
        rankconstant, rankquery = self.compute_parameters(query_indices)
        log_scores = rankconstant + self.dataset * rankquery.transpose()
        return np.asarray(log_scores.flatten())[0]

    def query_many(self, queries):
        rankconstants = []
        rankqueries = []
        for query in queries:
            rankconstant, rankquery = self.compute_parameters(query)
            rankconstants.append(rankconstant)
            rankqueries.append(rankquery)
        rankconstants = np.array(rankconstants)
        rankqueries = np.vstack(rankqueries)
        score_matrix = rankconstants + self.dataset * rankqueries.T
        return score_matrix.T

    def compute_parameters(self, query_indices):
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

    @staticmethod
    def estimate_hyperparameters(meanfactor, dataset):
        alpha = meanfactor * dataset.mean(0)
        beta = meanfactor - alpha
        return alpha, beta
