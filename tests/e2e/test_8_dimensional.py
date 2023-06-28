import numpy as np
import pandas as pd
import pytest
import scipy

from copula_wrapper import CopulaJoint
from tests.e2e import assert_kolmogorov_smirnov, assert_rank_corr


@pytest.fixture
def kendall_tau():
    """
    It's a big effort to manually come up with an 8x8 Kendall's tau matrix that is nontrivial,
    yet still satisfies the property that the corresponding Pearson's rho matrix used to
    parametrize the Gaussian copula is positive semi-definite. So this approach instead.
    """

    # Randomly generated using the onion method
    # See: https://github.com/tamaghnaroy/RandomCorrMat/blob/5b2f286e4e345a5bac0b3c20ed0fe4dfc5b9ef45/RandomCorrMat/RandomCorr.py#L37-L71
    corr = np.array(
        [
            [1.0, 0.622, -0.164, 0.489, -0.074, -0.159, 0.376, 0.382],
            [0.622, 1.0, -0.332, 0.417, -0.572, 0.263, 0.439, 0.566],
            [-0.164, -0.332, 1.0, -0.042, 0.465, -0.385, -0.159, -0.623],
            [0.489, 0.417, -0.042, 1.0, -0.45, 0.362, 0.353, 0.013],
            [-0.074, -0.572, 0.465, -0.45, 1.0, -0.688, 0.165, -0.454],
            [-0.159, 0.263, -0.385, 0.362, -0.688, 1.0, -0.191, 0.195],
            [0.376, 0.439, -0.159, 0.353, 0.165, -0.191, 1.0, 0.09],
            [0.382, 0.566, -0.623, 0.013, -0.454, 0.195, 0.09, 1.0],
        ]
    )

    data = scipy.stats.multivariate_normal.rvs(mean=np.zeros(8), cov=corr, size=10_000)

    kendall = np.eye(8)
    for i in range(8):
        for j in range(i + 1, 8):
            kendall[i, j] = scipy.stats.kendalltau(data[:, i], data[:, j]).correlation
            kendall[j, i] = kendall[i, j]

    return kendall


def test(kendall_tau):
    marginals = [
        scipy.stats.norm(0, 1),
        scipy.stats.lognorm(1, 1),
        scipy.stats.cauchy(0, 1),
        scipy.stats.gamma(1, 2),
        scipy.stats.beta(2, 2),
        scipy.stats.uniform(0, 5),
        scipy.stats.expon(1),
        scipy.stats.chi2(1),
    ]

    dist = CopulaJoint(marginals, kendall_tau=kendall_tau)

    # Get into right format for my helper functions, since this test is the odd one out
    # in using a list of marginals instead of a dict
    marginals = {i: marginals[i] for i in range(8)}
    joint_sample = pd.DataFrame(dist.rvs(1_000_000))
    kendall_tau = {(i, j): kendall_tau[i, j] for i in range(8) for j in range(i + 1, 8)}

    assert_rank_corr(joint_sample, kendall_tau, "kendall_tau", rel=1 / 100, abs=1 / 100)
    assert_kolmogorov_smirnov(joint_sample, marginals, tol=1 / 100)
