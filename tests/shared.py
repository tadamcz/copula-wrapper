import numpy as np
import pytest
from scipy import stats

from copula_wrapper.joint_distribution import JointDistribution


def is_frozen_normal(distribution):
	if isinstance(distribution, stats._distn_infrastructure.rv_frozen):
		if isinstance(distribution.dist, stats._continuous_distns.norm_gen):
			return True
	return False


def is_frozen_lognormal(distribution):
	if isinstance(distribution, stats._distn_infrastructure.rv_frozen):
		if isinstance(distribution.dist, stats._continuous_distns.lognorm_gen):
			return True
	return False


def is_frozen_beta(distribution):
	if isinstance(distribution, stats._distn_infrastructure.rv_frozen):
		if isinstance(distribution.dist, stats._continuous_distns.beta_gen):
			return True
	return False


@pytest.fixture(params=[-1, 0.5], ids=lambda p: f"mu_n={p}")
def mu_norm(request):
	return request.param


@pytest.fixture(params=[1], ids=lambda p: f"sigma_n={p}")
def sigma_norm(request):
	return request.param


@pytest.fixture(params=[0], ids=lambda p: f"mu_lg={p}")
def mu_lognorm(request):
	return request.param


@pytest.fixture(params=[1], ids=lambda p: f"sigma_lg={p}")
def sigma_lognorm(request):
	return request.param


@pytest.fixture(params=[2, 3], ids=lambda p: f"alpha={p}")
def alpha(request):
	return request.param


@pytest.fixture(params=[1], ids=lambda p: f"beta={p}")
def beta(request):
	return request.param


@pytest.fixture()
def marginals(mu_norm, sigma_norm, mu_lognorm, sigma_lognorm, alpha, beta):
	return {
		"n": stats.norm(mu_norm, sigma_norm),
		"l": stats.lognorm(scale=np.exp(mu_lognorm), s=sigma_lognorm),
		"b": stats.beta(alpha, beta),
	}


@pytest.fixture(params=[
	{("n", "l"): 0.2, ("n", "b"): 0.3, ("l", "b"): 0.6},
	{("n", "l"): 0.5, ("n", "b"): 0.5, ("l", "b"): 0.5},
	{("n", "l"): 0.99, ("n", "b"): 0.99, ("l", "b"): 0.99},
], ids=lambda rs: f"pairwise_corrs={rs}")
def rank_corr(request):
	return request.param


@pytest.fixture(params=['spearman', 'kendall'], ids=lambda p: f"method={p}")
def rank_corr_method(request):
	return request.param


@pytest.fixture()
def joint_distribution(marginals, rank_corr, rank_corr_method):
	return JointDistribution(marginals=marginals, rank_corr=rank_corr, rank_corr_method=rank_corr_method)


@pytest.fixture()
def joint_sample(joint_distribution):
	sample = joint_distribution.rvs(5_000_000)
	return sample
