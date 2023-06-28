import numpy as np
from rvtools import certainty
from scipy import stats

import tests.shared
from copula_wrapper import CopulaJoint
from tests.e2e import assert_kolmogorov_smirnov


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
    assert_kolmogorov_smirnov(joint_sample, marginals, tol=1 / 500)


def test_certainty(rank_corr_measure):
    VALUE = 1.23
    marginals = {
        "n": stats.norm(1, 1),
        "b": stats.beta(2, 3),
        "certainty": certainty(VALUE),
    }
    rank_corr = {("n", "certainty"): 0.42}
    jd = CopulaJoint(marginals, **{rank_corr_measure: rank_corr})
    joint_sample = jd.rvs(1_000_000)

    assert np.all(joint_sample["certainty"] == VALUE)

    del marginals["certainty"]  # This won't work with Kolmogorov-Smirnov
    assert_kolmogorov_smirnov(joint_sample, marginals, tol=1 / 500)
