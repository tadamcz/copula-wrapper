Suppose we want to specify a continuous `n`-dimensional joint probability distribution by giving:

- `n` arbitrary marginal distributions
- a set of pairwise correlations between the dimensions

[Copulas](https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html) are a good way to achieve this separate specification of marginal distribution and dependence.

This is a wrapper around the ``CopulaDistribution`` class from the [`statsmodels` package](https://www.statsmodels.org/), providing an interface that I prefer. (The rationale is disused in more detail [below](#discussion-of-interface)).
# Installation

```shell
pip install copula_wrapper  # or `poetry add copula_wrapper`
```
# Usage

Dictionary arguments and Kendall's tau:

```python
from copula_wrapper import CopulaJoint
from scipy import stats
import seaborn as sns

marginals = {
    "consumption elasticity": stats.uniform(0.75, 3),
    "market return": stats.lognorm(0.05, 0.05),
    "risk of war": stats.beta(1, 50)
}

tau = {
    # Missing pairs are assumed to be independent
    ("risk of war", "market return"): -0.5,
}

# Instantiate the (frozen) joint distribution
dist = CopulaJoint(marginals, kendall_tau=tau)

# Query the CDF or PDF
p = dist.cdf({
    "consumption elasticity": 1.5,
    "market return": 1.5,
    "risk of war": 0.5
})

# Sample randomly
sample = dist.rvs(100_000)

# Plot
sns.pairplot(sample, plot_kws={"s": 1})
```
![](plot1.png)

Array arguments and Spearman's rho:

```python
import pandas as pd
import numpy as np
from copula_wrapper import CopulaJoint
from scipy import stats
import seaborn as sns

marginals = [
    stats.uniform(0.75, 3),
    stats.lognorm(0.05, 0.05),
    stats.beta(1, 50)
]

spearman = np.eye(3)
spearman[1, 2] = spearman[2, 1] = -0.5

# Instantiate the (frozen) joint distribution
dist = CopulaJoint(marginals, spearman_rho=spearman)

# Query the CDF or PDF
p = dist.cdf([1.5, 1.5, 0.5])

# Sample randomly
sample = dist.rvs(10_000)

# Plot
sns.pairplot(pd.DataFrame(sample), plot_kws={"s": 1})
```
![img_2.png](plot2.png)

# Discussion of interface

``CopulaJoint`` is a wrapper around ``CopulaDistribution`` from ``statsmodels``.

The context that motivated this wrapper was:
- A Monte Carlo simulation model with 10+ input dimensions
- The correlations come from subjective judgments / expert elicitation
- Many correlations are zero
- Users may not be familiar with or care about the mathematics of copulas (I wasn't) and may not be familiar with API conventions of SciPy (that also influenced `statsmodels`).
- In my opinion, the terminology in the area invites confusion between a copula and the joint distribution (with arbitrary marginals) that it induces.

## How this wrapper changes the API

### Abstracts away the underlying copula and takes in rank correlations
In ``statsmodels``, you'd do something like the following

```python
from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula
from scipy import stats
corr = [
	[1.0, 0.5],
	[0.5, 1.0],
]
joint = CopulaDistribution(GaussianCopula(corr=corr), marginals=[stats.beta(2,2), stats.norm()])
```

where ``corr`` is the matrix of Pearson's rho correlations (the standard linear correlation coefficient most people are familiar with). IMO, this surfaces implementation details and requires the user to engage with the internals of their joint distribution, which they may not understand or care about. It also invites the misunderstanding that the joint distribution will have a Pearson's rho matching the ``corr`` we pass in. This is not the case, as the following example shows:

```python
import numpy as np
from scipy import stats
from statsmodels.distributions.copula.api import CopulaDistribution, GaussianCopula

c = 0.8
corr = [
    [1.0, c],
    [c, 1.0],
]
joint = CopulaDistribution(GaussianCopula(corr=corr),
                           marginals=[stats.norm(2, 2), stats.lognorm(loc=0, s=1)])

samples = joint.rvs(int(1e6))
joint_r = stats.pearsonr(samples[:, 0], samples[:, 1])[0]  # about 0.6
assert not np.isclose(joint_r, c)
```

Therefore, this wrapper instead takes rank correlations as input, and these are satisfied by the resulting joint distribution. It converts these to the appropriate correlation measure for the underlying copula. You must explicitly provide ``kendall_tau`` or ``spearman_rho``.

(For now, only the Gaussian copula is supported. But I've written the class to be extensible easily and even put some hints as code comments. PRs welcome. If more copulas are supported, there will be a parameter ``family`` with values like ``"Gaussian"``, ``"Clayton"``...)

### Allows dimension names
if dimensions are to be named, ``marginals`` must be a dict, and ``rank_corr`` must be a dict
of tuples, e.g.

```python
from scipy import stats
marginals = {
	"a": stats.uniform(0, 1),
	"b": stats.lognorm(2, 3),
	"c": stats.norm(4, 5),
}

rank_corr = {
	# Missing pairs are assumed to be independent
	("a", "b"): 0.5,
}
```

and the first argument to ``.pdf``, ``.cdf``, and ``.logpdf`` must be a dict with the same keys as ``marginals``.
(I originally had these methods support dimensions as keyword arguments, but I thought it was better not to
modify the API even more, plus dictionary keys can be any hashable object, not just strings.).

If dimensions are named,
``.rvs`` returns a Pandas DataFrame with dimension names as column names. 

Otherwise, ``marginals`` must be a list and ``rank_corr`` must be a square matrix, and ``.rvs``
returns a numpy array, just like in statsmodels.

### Satisfies interface of SciPy multivariate frozen distribution
For background, the Scipy pattern is that there is e.g. a ``norm_gen`` subclass of ``rv_continuous`` that is used to create the
``norm`` instance, which can then optionally be 'frozen'.

The statsmodels ``CopulaDistribution`` does not follow this pattern. It acts 'frozen' in sofar
as the underlying copula (e.g. Gaussian) and its correlation matrix are specified at
initialization. However, it acts 'unfrozen' in that ``cop_args`` and ``marg_args`` can
be passed to ``.pdf``, ``.cdf``, and ``.logpdf``. As a result, ``CopulaDistribution`` does not subclass
any of the ``scipy.stats`` classes.

I find this a bit inconsistent, and don't find ``cop_args`` and ``marg_args`` useful, so I
drop them in this wrapper.	The benefit is that the wrapper now satisfies the interface of
a SciPy multivariate frozen distribution.

This means that the marginals provided to ``CopulaJoint`` must be frozen as well.

It also means I use `size` (SciPy) instead of `nobs` (statsmodels) for the call to `rvs`. 

# Tests

While the copula tests in `statsmodels` are likely to be good for their intended purpose, I couldn't quickly decipher
exactly what behaviour these tests were checking for. In order to make sure this wrapper behaves in the way that its
interface promises, I created a set of end-to-end tests. These tests that check that random samples from the
joint distribution have:

1. The expected rank correlation matrix
2. The expected marginal distributions (using the Kolmogorov-Smirnov statistic)

The tests are parametrized (run for many inputs). 

As they involve large numbers of random samples, they are slow.

There are also a few simple unit tests.
