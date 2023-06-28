import numpy as np
import pytest
from scipy import stats

from copula_wrapper import CopulaJoint


def test_inconsistent_dict(rank_corr_measure):
    marginals = {
        "a": stats.norm(1, 1),
        "b": stats.norm(1, 1),
        "c": stats.norm(1, 1),
    }
    rank_corr = {
        ("a", "b"): 0.5,
        ("b", "a"): 0.1,
    }
    with pytest.raises(ValueError, match="Inconsistent rank correlation"):
        CopulaJoint(marginals, **{rank_corr_measure: rank_corr})


def test_inconsistent_matrix(rank_corr_measure):
    marginals = [
        stats.norm(1, 1),
        stats.norm(1, 1),
        stats.norm(1, 1),
    ]
    rank_corr = [
        [1, 0.1, 0],
        [0.5, 1, 0],
        [0, 0, 1],
    ]
    with pytest.raises(ValueError, match="Inconsistent rank correlation"):
        CopulaJoint(marginals, **{rank_corr_measure: rank_corr})


def test_not_positive_semidefinite(rank_corr_measure):
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
    with pytest.raises(ValueError, match="positive semidefinite"):
        CopulaJoint(marginals, **{rank_corr_measure: rank_corr})


@pytest.mark.parametrize("allow_singular", [True, False])
def test_allow_singular(allow_singular):
    marginals = {
        "a": stats.uniform(0.75, 3),
        "b": stats.uniform(0.75, 3),
        "c": stats.norm(1, 1),
    }
    tau = {
        ("a", "b"): 1,
    }
    if allow_singular:
        joint = CopulaJoint(marginals, kendall_tau=tau, allow_singular=allow_singular)
    else:
        with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
            joint = CopulaJoint(marginals, kendall_tau=tau, allow_singular=allow_singular)
