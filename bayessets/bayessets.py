import numpy as np
from scipy.sparse import csr_matrix


class BayesianSet:
    def __init__(self, dataset, alpha, beta):
        self.X = csr_matrix(dataset.T)
        self.alpha = alpha
        self.beta = beta
        self.alpha_plus_beta = self.alpha + self.beta
        self.log_alpha = np.log(self.alpha)
        self.log_beta = np.log(self.beta)
        self.log_alpha_plus_beta = np.log(self.alpha_plus_beta)

    def query(self, query_indices):
        c, q = self.compute_parameters(query_indices)
        log_scores = c + self.X * q.transpose()
        return np.asarray(log_scores.flatten())[0]

    def query_many(self, queries):
        c = []
        q = []
        for query in queries:
            ci, qi = self.compute_parameters(query)
            c.append(ci)
            q.append(qi)
        c = np.array(c)
        q = np.vstack(q)
        score_matrix = c + self.X * q.T
        return score_matrix.T

    def compute_parameters(self, query_indices):
        N = len(query_indices)
        sum_x = np.sum(self.X[query_indices], axis=0)
        alpha_tilde = sum_x + self.alpha
        beta_tilde = self.beta + N - sum_x
        log_alpha_tilde = np.log(alpha_tilde)
        log_beta_tilde = np.log(beta_tilde)
        c = (self.alpha_plus_beta - np.log(self.alpha + self.beta + N) +
             log_beta_tilde - self.log_beta).sum()
        q = log_alpha_tilde - self.log_alpha - log_beta_tilde + self.log_beta
        return c, q

    @staticmethod
    def estimate_hyperparameters(c, dataset):
        alpha = c * csr_matrix(dataset.T).mean(0)
        beta = c - alpha
        return alpha, beta
