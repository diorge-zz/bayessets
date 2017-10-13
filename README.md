# Bayesian Sets

Implementation of the Bayesian Sets algorithm and variants.
Bayesian Sets is a set expansion algorithm for generic datasets.
The original paper [1] proposes a model for the exponential family of distributions,
and with greater detail a model using the independent Bernoulli distribution.

The repository is available on PyPI and can be installed with `pip install bayessets`.
The complete documentation can be found at [ReadTheDocs](http://bayessets.readthedocs.io/en/latest/).

## Basic usage

After you create your model, you can use the `query` method,
passing the index of your seed set, to get the score vector for every instance.
If you want the index of the most likely expansion,
use `numpy.argsort` and reverse it, as shown below.

```python
import bayessets
import numpy as np

model = bayessets.BernoulliBayesianSet(mybinarymatrix)
myquery = [23, 50, 78] # 0-based index of seed set
scores = model.query(myquery)
ranking = np.argsort(scores)[::-1]
top20 = ranking[:20]
```

## References
[1] Ghahramani, Zoubin, and Katherine A. Heller. "Bayesian sets." Advances in neural information processing systems (2006): 435-442.
