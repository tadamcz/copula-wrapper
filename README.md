Suppose we want to specify a continuous `n`-dimensional joint probability distribution by giving:

- `n` marginal distributions
- a set of pairwise correlations between the dimensions

[Copulas](https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html) are a good way to achieve this separate specification of dependence and marginal distribution.

This is a wrapper around the copula functionality in the [`statsmodels` package](https://www.statsmodels.org/).

# Interface

The `statsmodels` package has a good set of functionality related to copulas, found
in `statsmodels.distributions.copula`. However, for my purposes, I found the `statsmodels` interface unintuitive.

This wrapper abstracts away copula-related considerations:

* It takes as inputs:
    * the marginal distributions (as SciPy continuous distributions objects)
    * optional [rank correlations](https://en.wikipedia.org/wiki/Rank_correlation) (note, this is not the traditional [Pearson's correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)).
* It offers a standard SciPy distribution interface (full SciPy compatibility is a work in progress, see TODO in code, PR welcome).

The wrapper currently only supports the Gaussian copula.

For more detail, see docstrings.

# Tests

While the copula tests in `statsmodels` are likely to be good for their intended purpose, I couldn't quickly decipher
exactly what behaviour these tests were checking for. In order to make sure this wrapper behaves in the way that its
interface implicitly promises, I created a set of end-to-end tests. These tests that check that random samples from the
joint distribution have:

1. The expected rank correlation matrix
2. The expected marginal distributions (using the Kolmogorov-Smirnov statistic)

The tests are parametrized (run for many inputs). If you have multiple cores, it's recommended that you run tests in
parallel with `pytest -n <x>` (where `x` is the number of processes) or `pytest -n auto`.

(There is also one very simple unit test for the transformation of correlations).

# Usage

```python
from copula_wrapper import JointDistribution
from scipy import stats

marginals = {
	"consumption_elasticity": stats.uniform(0.75, 3),
	"market_return": stats.lognorm(1, 1),
	"risk_of_war": stats.beta(1, 100)
}

pairwise_rank_corr = {
	# Missing pairs are assumed to be independent
	("risk_of_war", "market_return"): -0.1,
}

dist = JointDistribution(marginals, pairwise_rank_corr, rank_corr_method="kendall")

# Query the CDF or PDF
p = dist.cdf(
	consumption_elasticity=1,
	market_return=5,
	risk_of_war=0.005,
)

# Sample randomly
sample = dist.rvs(100_000)
```
