import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.distributions.copula.api import GaussianCopula
from statsmodels.distributions.copula.copulas import CopulaDistribution

from copula_wrapper import correlation_convert


class JointDistribution(scipy.stats.rv_continuous):
	"""
	Thin wrapper around `CopulaDistribution` from `statsmodels`.

	Currently, this exclusively uses the Gaussian copula.

	The main differences are:
	- This interface explicitly requires a rank correlation, instead of the Pearson's rho of the transformed variables.
	- Dimensions require names (given as dictionary keys) and must be accessed by their names (as keyword arguments to
	.pdf, .cdf, and .logpdf), instead of their indices.
	- .rvs returns samples as a Pandas DataFrame with dimension names as column names.

	todo:
		- improve subclassing of scipy.rv_continuous. Current approach is quick and dirty, and fails for some
		methods that should work, like .ppf. This is because you're supposed to override the underscored methods ._pdf,
		._cdf, etc., instead of the methods .pdf, .cdf, etc. I haven't yet figured out how override these well for an
		n-dimensional distribution.
	"""

	def __init__(self, marginals, rank_corr, rank_corr_method):
		"""
		:param marginals: Dictionary of size `n`, where the keys are dimension names as strings and the values are
		SciPy continuous distributions.

		:param rank_corr: Dictionary of pairwise rank correlations. Missing pairs are
		assumed to be independent.

		:param rank_corr_method: 'spearmans_rho' or 'kendalls_tau'
		"""
		super().__init__()
		self.marginals = marginals
		self.rank_correlation = rank_corr
		self.rank_correlation_method = rank_corr_method

		self.dimension_names = {}
		marginals_list = [None] * len(marginals)
		for index, (name, distribution) in enumerate(marginals.items()):
			self.dimension_names[name] = index
			marginals_list[index] = distribution

		rank_corr_matrix = self._to_matrix(rank_corr)

		if rank_corr_method == 'spearmans_rho':
			pearsons_rho = correlation_convert.pearsons_rho(spearmans_rho=rank_corr_matrix)
		elif rank_corr_method == 'kendalls_tau':
			pearsons_rho = correlation_convert.pearsons_rho(kendalls_tau=rank_corr_matrix)
		else:
			raise ValueError("`rank_corr_method` must be one of 'spearmans_rho' or 'kendalls_tau'")

		# `pearsons_rho` refers to the correlations of the Gaussian-transformed variables
		copula_instance = GaussianCopula(corr=pearsons_rho)
		self.wrapped = CopulaDistribution(copula_instance, marginals_list)
		self.wrapped.rank_correlation = rank_corr_matrix

	def rvs(self, nobs=2, random_state=None):
		as_df = pd.DataFrame()
		rvs = self.wrapped.rvs(nobs=nobs, random_state=random_state)
		for name, i in self.dimension_names.items():
			column = rvs[:, i]
			as_df[name] = column
		return as_df

	def cdf(self, **kwargs):
		return self.wrapped.cdf(self._to_tuple(kwargs))

	def pdf(self, **kwargs):
		return self.wrapped.pdf(self._to_tuple(kwargs))

	def logpdf(self, **kwargs):
		return self.wrapped.logpdf(self._to_tuple(kwargs))

	def sf(self, **kwargs):
		return 1 - self.cdf(**kwargs)

	def _to_tuple(self, kwargs):
		if kwargs.keys() != self.dimension_names.keys():
			raise ValueError(f"You must provide the following keyword arguments: {list(self.dimension_names.keys())}")
		iterable = [None] * len(self.marginals)
		for name, index in self.dimension_names.items():
			iterable[index] = kwargs[name]
		return tuple(iterable)

	def _to_matrix(self, rank_correlation):
		corr_matrix = np.eye(N=len(self.marginals))
		names_to_indices = self.dimension_names
		for index, (pair, correlation) in enumerate(rank_correlation.items()):
			left, right = pair
			i, j = names_to_indices[left], names_to_indices[right]
			corr_matrix[i][j] = correlation
			corr_matrix[j][i] = correlation
		return corr_matrix
