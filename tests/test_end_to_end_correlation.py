import itertools

import numpy as np
import pandas as pd
import pytest
import scipy
import scipy.linalg
from scipy import stats


def test_rank_correlation(joint_sample, rank_corr, rank_corr_method):
	rank_correlation_helper(joint_sample, rank_corr, rank_corr_method, rel=1 / 100, abs=0)


def rank_correlation_helper(joint_sample, corr, rank_correlation_method, rel, abs):
	sample_corr = joint_sample.corr(method=rank_correlation_method)

	for key, value in corr.items():
		left, right = key
		assert sample_corr[left][right] == pytest.approx(value, rel=rel, abs=abs)


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
	r = pd.DataFrame(samples_avg).corr(method=rank_correlation_method)
	r = to_corr_dict(r)
	return r


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
