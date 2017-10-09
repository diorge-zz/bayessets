Getting Started
===============

Bayesian Sets is a machine learning algorithm
for the set expansion learning task,
using arbitrary data types.
"Set expansion" refers to when you have a large set of items
(like all the words in the English dictionary),
and you know a subset of it that share some characteristics,
for example a few names of fruits.
The task of expanding the set is finding other items
that also share the same characteristics.

To install ``bayessets``, simply use ``pip install bayessets``.
The libraries Numpy and Scipy are required
for efficiently handling of sparse matrices,
and will be automatically installed.

The Wine dataset
----------------

Set expansion is not suitable for classification tasks,
but we will use here a very simple dataset built for classification,
for illustrative purposes.
Here we will be using the Wine dataset from UCI.
Scikit Learn will be required for some preprocessing and importing the data. ::

   import bayessets
   import numpy as np
   import sklearn.datasets
   from sklearn.preprocessing import Normalizer, Binarizer

   wine = sklearn.datasets.load_wine()

We will be using the :py:meth:`BernoulliBayesianSet`,
so we need to convert the data to binary.
Here, we simply use the mean of each feature to binarize,
after normalizing to the [0, 1] range. ::

   normalised = Normalizer().fit_transform(wine.data)
   nsample, nfeatures = normalised.shape
   bindata = np.zeros(normalised.shape)
   for i in range(nfeatures):
      binarizer = Binarizer(threshold=normalised[:, i].mean())
      bindata[:, i] = binarizer.fit_transform(
                        normalised[:, i].reshape(-1, 1)
                      ).reshape(1, -1)

Now that we have our binary data,
we can train a model.
Here we will assume the hyper-parameters
to be two times the mean. ::

   model = BernoulliBayesianSet(bindata, meanfactor=2,
                                alphaepsilon=0.0001, betaepsilon=0.0001)

Now we can query our model to find expansion to our query set.
For example, we will take items with indices 0, 3 and 5,
which are all of class zero in the Wine dataset.
The ``argsort`` is used to transform the scores back to indices,
in descending order (``[::-1``]). ::

   myquery = [0, 3, 5]
   ranking = np.argsort(model.query(myquery))[::-1]
   top10 = ranking[:10]
   classes = wine.target[top10]
   hits = (classes == 0).sum()
   precision = hits / 10.0
   print(precision)
