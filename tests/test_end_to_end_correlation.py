import itertools
import string

import pandas as pd
import pytest
import scipy.linalg
from scipy import stats

from copula_wrapper import JointDistribution
from tests.shared import rand_corr_onion
from tests.test_end_to_end_marginals import kolmogorov_smirnov_helper


def test_rank_correlation(joint_sample, rank_corr, rank_corr_method):
	rank_correlation_helper(joint_sample, rank_corr, rank_corr_method, rel=1 / 100, abs=0)


def rank_correlation_helper(joint_sample, corr, rank_correlation_method, rel, abs):
	sample_corr = joint_sample.corr(method=method_pandas(rank_correlation_method))

	for key, value in corr.items():
		left, right = key
		assert sample_corr[left][right] == pytest.approx(value, rel=rel, abs=abs)


def method_pandas(method):
	# todo: remove this and the associated application code. better to follow the convention set down by pandas.
	if method == 'spearmans_rho':
		return 'spearman'
	elif method == 'kendalls_tau':
		return 'kendall'


def to_corr_dict(dataframe):
	pairs = itertools.combinations(dataframe.columns, r=2)
	corr_dict = {}
	for pair in pairs:
		left, right = pair
		corr_dict[(left, right)] = dataframe[left][right]
	return corr_dict


def random_rank_correlation(marginals, rank_correlation_method):
	"""
	Todo: figure out a better way to make this generator reliable
	"""
	random_pearson_rho = rand_corr_onion(len(marginals))
	sample_1 = scipy.stats.multivariate_normal(cov=random_pearson_rho).rvs(100)
	random_pearson_rho = rand_corr_onion(len(marginals))
	sample_2 = scipy.stats.multivariate_normal(cov=random_pearson_rho).rvs(100)
	samples_avg = sample_1 + sample_2
	samples_avg = pd.DataFrame(samples_avg, columns=marginals.keys())
	r = pd.DataFrame(samples_avg).corr(method=method_pandas(rank_correlation_method))
	r = to_corr_dict(r)
	return r


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
