import numpy as np
from nonstd.distributions import FrozenCertainty
from scipy import stats

import tests.shared
from copula_wrapper import JointDistribution


def test_domain(joint_sample, marginals):
	for name, marginal in marginals.items():
		data = joint_sample[name]
		if tests.shared.is_frozen_lognormal(marginal):
			assert np.all(data > 0)
		if tests.shared.is_frozen_beta(marginal):
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


def test_certainty(rank_corr_method):
	VALUE = 1.23
	marginals = {
		"n": stats.norm(1, 1),
		"b": stats.beta(2, 3),
		"certainty": FrozenCertainty(VALUE),
	}
	rank_corr = {("n", "certainty"): 0.42}
	jd = JointDistribution(marginals, rank_corr, rank_corr_method)
	joint_sample = jd.rvs(nobs=1_000_000)

	assert np.all(joint_sample["certainty"] == VALUE)

	del marginals["certainty"]  # This won't work with Kolmogorov-Smirnov
	kolmogorov_smirnov_helper(joint_sample, marginals, tol=1 / 500)
