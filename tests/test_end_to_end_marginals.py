import numpy as np
from scipy import stats

from tests import shared


def test_domain(joint_sample, marginals):
	for name, marginal in marginals.items():
		data = joint_sample[name]
		if shared.is_frozen_lognormal(marginal):
			assert np.all(data > 0)
		if shared.is_frozen_beta(marginal):
			assert np.all(0 < data) and np.all(data < 1)


def test_kolmogorov_smirnov(joint_sample, marginals):
	"""
	The Kolmogorov-Smirnov statistic is the	maximum vertical distance between the empirical CDF and the theoretical CDF.

	It's therefore between 0 and 1.
	"""
	kolmogorov_smirnov_helper(joint_sample, marginals, tol=1 / 500)


def kolmogorov_smirnov_helper(joint_sample, marginals, tol):
	subsample_n = 1_000_000
	for name, marginal in marginals.items():
		data = joint_sample[name]
		# Since computing the K-S statistic is expensive, we just take a sufficiently large subset of data from the fixture
		data = data[:subsample_n]
		ks_statistic = stats.kstest(data, marginal.cdf).statistic
		assert ks_statistic < tol
