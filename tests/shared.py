import numpy as np
import pytest
import scipy
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


@pytest.fixture(params=['spearmans_rho', 'kendalls_tau'], ids=lambda p: f"method={p}")
def rank_corr_method(request):
	return request.param


@pytest.fixture()
def joint_distribution(marginals, rank_corr, rank_corr_method):
	return JointDistribution(marginals=marginals, rank_corr=rank_corr, rank_corr_method=rank_corr_method)


@pytest.fixture()
def joint_sample(joint_distribution):
	sample = joint_distribution.rvs(5_000_000)
	return sample


def rand_corr_onion(size):
	"""
	Adapted from Tamaghna Roy:
	https://github.com/tamaghnaroy/RandomCorrMat/blob/master/RandomCorrMat/RandomCorr.py

	This algorithm samples exactly and very quickly from a uniform distribution over the space of correlation matrices.
	The idea here is to build the correlation matrix recursively
		Corr(dimension=d) = [Corr(dimension=d-1) q; q 1]
	q is chosen by the algorithm to ensure that it is a valid correlation matrix

	original paper: https://people.orie.cornell.edu/shane/pubs/NORTAHighD.pdf

	matlab code: https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices/125017#125017

	@param size: size of the correlation matrix
	@return: correlation matrix (size x size)
	"""
	S = [[1]]
	for i in range(1, size):
		k = i + 1
		if k == size:
			y = np.random.uniform(0, 1)
		else:
			y = np.random.beta((k - 1) / 2, (size - k) / 2)
		r = np.sqrt(y)
		theta = np.random.randn(k - 1, 1)
		theta = theta / np.sqrt(np.dot(theta.T, theta))
		w = r * theta
		R = scipy.linalg.sqrtm(S)
		q = np.dot(R, w)
		S = np.vstack((np.hstack((S, q)), np.hstack((q.T, [[1]]))))
	return S
