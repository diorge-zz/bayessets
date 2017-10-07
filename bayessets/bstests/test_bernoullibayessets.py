"""Unit tests for the Bernoulli Bayesian Set algorithm
"""


from __future__ import division
from bayessets import BernoulliBayesianSet
import numpy as np
import sklearn.datasets
from sklearn.preprocessing import Normalizer, Binarizer


def test_hyperparameters_from_mean():
    """Tests the calculation of hyper-parameters alpha and beta
    """
    data = np.array([[1, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 1],
                     [0, 1, 0, 0]])
    means = np.array([0.25, 0.5, 0.25, 0.75])
    scale = 2
    model = BernoulliBayesianSet(data, meanfactor=scale)
    assert np.all(model.alpha == (scale * means))
    assert np.all(model.beta == scale - model.alpha)


def test_hyperparameters_from_mean_unusual_scale():
    """Tests the calculation of hyper-parameters alpha and beta,
    not using the default C=2 constant
    """
    data = np.array([[0, 0, 0, 1],
                     [0, 1, 0, 1],
                     [0, 0, 1, 1],
                     [1, 1, 0, 0]])
    means = np.array([0.25, 0.5, 0.25, 0.75])
    scale = 15
    model = BernoulliBayesianSet(data, scale)
    alpha, beta = model.alpha, model.beta
    assert np.all(alpha == (scale * means))
    assert np.all(beta == scale - alpha)


def test_wine():
    """Sample test on the Wine UCI dataset.

    Please do note this test is _not_ conclusive,
    but the zero class is so well-separated
    that all the variations should do well on this
    specific class
    """
    wine = sklearn.datasets.load_wine()
    train_data = Normalizer().fit_transform(wine.data)
    nsample, nfeatures = train_data.shape
    bindata = np.zeros(train_data.shape)
    for i in range(nfeatures):
        binarizer = Binarizer(threshold=train_data[:, i].mean())
        bindata[:, i] = binarizer.fit_transform(train_data[:, i]
                                                .reshape(-1, 1)).reshape(1, -1)
    model = BernoulliBayesianSet(bindata, meanfactor=2,
                                 alphaepsilon=0.0001, betaepsilon=0.0001)
    some_zero_class_indices = [0, 3, 5]
    ranking = np.argsort(model.query(some_zero_class_indices))[::-1]
    top10 = ranking[:10]
    truepositives = (wine.target[top10] == 0).sum()
    precision = truepositives / 10
    # allows a single mistake
    assert precision >= 0.9


def test_query_many():
    """Test for the query_many method
    """
    data = np.random.randn(20, 6)
    bindata = Binarizer(threshold=0.5).fit_transform(data)
    model = BernoulliBayesianSet(bindata)
    individual = []
    queries = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    for query in queries:
        individual.append(model.query(query))
    total = model.query_many(queries)
    for rank in individual:
        assert rank in total
