import string

from scipy import stats

from copula_wrapper import JointDistribution
from tests.test_end_to_end_correlation import random_rank_correlation, rank_correlation_helper
from tests.test_end_to_end_marginals import kolmogorov_smirnov_helper


def test_many_dimensions(rank_corr_method):
	marginals = [
		stats.norm(0, 1),
		stats.lognorm(1, 1),
		stats.cauchy(0, 1),
		stats.gamma(1, 2),
		stats.beta(2, 2),
		stats.uniform(0, 5),
		stats.expon(1),
		stats.chi2(1),
	]
	alphabet = string.ascii_lowercase
	marginals = {alphabet[i]: v for i, v in enumerate(marginals)}

	rank_correlation = random_rank_correlation(marginals, rank_corr_method)

	dist = JointDistribution(marginals=marginals, rank_corr=rank_correlation, rank_corr_method=rank_corr_method)
	joint_sample = dist.rvs(1_000_000)
	rank_correlation_helper(joint_sample, rank_correlation, rank_corr_method, rel=1 / 100, abs=1 / 100)
	kolmogorov_smirnov_helper(joint_sample, marginals, tol=1 / 100)
