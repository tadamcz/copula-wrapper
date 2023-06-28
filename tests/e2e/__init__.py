import pytest
import scipy


def assert_rank_corr(joint_sample, corr: dict, rank_correlation_method, rel, abs):
    """
    Helper when dealing with named dimensions
    """
    # Get into the right format for pandas / numpy
    if rank_correlation_method == "spearman_rho":
        rank_correlation_method = "spearman"
    elif rank_correlation_method == "kendall_tau":
        rank_correlation_method = "kendall"

    sample_corr = joint_sample.corr(method=rank_correlation_method)

    for key, value in corr.items():
        left, right = key
        assert sample_corr[left][right] == pytest.approx(value, rel=rel, abs=abs)


def assert_kolmogorov_smirnov(joint_sample, marginals, tol):
    """
    Helper when dealing with named dimensions
    """
    for name, marginal in marginals.items():
        data = joint_sample[name]
        ks_statistic = scipy.stats.kstest(data, marginal.cdf).statistic
        assert ks_statistic < tol
