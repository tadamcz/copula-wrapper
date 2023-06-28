import numpy as np


def to_pearsons_rho(kendall=None, spearman=None):
    """
    Bivariate Pearson's rho from bivariate rank correlations (Kendall's tau or Spearman's rho).

    References: https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html
    """

    if kendall is not None and spearman is not None:
        raise ValueError("Must provide exactly one of `kendall` or `spearman`.")

    if kendall is not None:
        func = lambda kendalls_tau: np.sin(kendalls_tau * np.pi / 2)
        arg = kendall
    elif spearman is not None:
        func = lambda spearmans_rho: 2 * np.sin(spearmans_rho * np.pi / 6)
        arg = spearman
    else:
        raise ValueError("Must provide exactly one of `kendall` or `spearman`.")

    try:
        return func(arg)
    except TypeError:
        return np.vectorize(func)(arg)
