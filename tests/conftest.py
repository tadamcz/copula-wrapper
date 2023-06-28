import os

import numpy as np
import pytest
from scipy import stats

import tests.seeds as seeds

n_random_seeds = int(os.environ.get("N_RAND_SEED", 3))


@pytest.fixture(
    params=seeds.RANDOM_SEEDS[:n_random_seeds],
    ids=lambda p: f"seed={p}",
)
def random_seed(request):
    np.random.seed(request.param)


from copula_wrapper.joint_distribution import CopulaJoint


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


@pytest.fixture
def marginals(mu_norm, sigma_norm, mu_lognorm, sigma_lognorm, alpha, beta):
    return {
        "n": stats.norm(mu_norm, sigma_norm),
        "l": stats.lognorm(scale=np.exp(mu_lognorm), s=sigma_lognorm),
        "b": stats.beta(alpha, beta),
    }


@pytest.fixture(
    params=[
        {("n", "l"): 0.2, ("n", "b"): 0.3, ("l", "b"): 0.6},
        {("n", "l"): 0.5, ("n", "b"): 0.5, ("l", "b"): 0.5},
        {("n", "l"): 0.99, ("n", "b"): 0.99, ("l", "b"): 0.99},
    ],
    ids=lambda rs: f"pairwise_corrs={rs}",
)
def rank_corr(request):
    return request.param


@pytest.fixture(params=["spearman_rho", "kendall_tau"], ids=lambda p: f"rank_corr_measure={p}")
def rank_corr_measure(request):
    return request.param


@pytest.fixture
def joint_distribution(marginals, rank_corr, rank_corr_measure):
    return CopulaJoint(marginals, **{rank_corr_measure: rank_corr})


@pytest.fixture()
def joint_sample(joint_distribution, random_seed):
    sample = joint_distribution.rvs(500_000)
    return sample
