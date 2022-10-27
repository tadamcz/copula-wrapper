import pytest
from scipy import stats

from copula_wrapper import JointDistribution


def test_inconsistent_dict():
	marginals = {
		"a": stats.norm(1, 1),
		"b": stats.norm(1, 1),
		"c": stats.norm(1, 1),
	}
	rank_corr = {
		("a", "b"): 0.5,
		("b", "a"): 0.1,
	}
	with pytest.raises(ValueError, match="Inconsistent rank correlations"):
		JointDistribution(marginals, rank_corr, "kendall")


def test_not_positive_semidefinite(rank_corr_method):
	marginals = {
		"a": stats.norm(1, 1),
		"b": stats.norm(1, 1),
		"c": stats.norm(1, 1),
	}
	rank_corr = {
		("a", "b"): 0.8,
		("b", "c"): 0.8,
		("c", "a"): -0.5,
	}
	with pytest.raises(ValueError, match="the input matrix must be positive semidefinite"):
		JointDistribution(marginals, rank_corr, rank_corr_method)
