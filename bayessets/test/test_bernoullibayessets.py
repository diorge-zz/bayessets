"""Unit tests for the Bernoulli Bayesian Set algorithm
"""


from bayessets import BayesianSet
import numpy as np


def test_hyperparameters_from_mean():
    """Tests the calculation of hyper-parameters alpha and beta
    """
    data = np.array([[0, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 1],
                     [0, 1, 0, 1]])
    means = np.array([0, 0.5, 0.25, 1])
    scale = 2
    alpha, beta = BayesianSet.estimate_hyperparameters(scale, data)
    assert np.all(alpha == (scale * means))
    assert np.all(beta == scale - alpha)


def test_hyperparameters_from_mean_unusual_scale():
    """Tests the calculation of hyper-parameters alpha and beta,
    not using the default C=2 constant
    """
    data = np.array([[0, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 1],
                     [0, 1, 0, 1]])
    means = np.array([0, 0.5, 0.25, 1])
    scale = 15
    alpha, beta = BayesianSet.estimate_hyperparameters(scale, data)
    assert np.all(alpha == (scale * means))
    assert np.all(beta == scale - alpha)
