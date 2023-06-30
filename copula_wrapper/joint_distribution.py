from numbers import Real
from operator import xor
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.distributions.copula.api
from scipy.stats.distributions import rv_frozen
from statsmodels.distributions.copula.copulas import CopulaDistribution

from copula_wrapper.correlation_convert import to_pearsons_rho


class CopulaJoint:
    """
    Wrapper for ``CopulaDistribution`` from ``statsmodels``.

    :param marginals:
        A dictionary mapping names to marginal distributions, or a list of marginal distributions.

    :param spearman_rho:
        A dictionary mapping pairs of names to Spearman's rho rank correlation coefficients,
        or a matrix of such rank correlations.

    :param kendall_tau:
        A dictionary mapping pairs of names to Kendall's tau rank correlation coefficients,
        or a matrix of such rank correlations.

    :param allow_singular:
        Whether to allow singular rank correlation matrices. The behavior when the correlation
        matrix is singular is determined by  ``scipy.stats.multivariate_normal`` and might not
        be appropriate for all methods. Behavior might change in future versions.

    Provide exactly one of ``spearman_rho`` or ``kendall_tau``.

    All marginal distributions must be frozen, i.e. have their parameters specified.

    Dictionary keys may be any hashable object, not just strings.


    Examples
    --------

    >>> from copula_wrapper import CopulaJoint
    >>> from scipy import stats
    >>>
    >>> marginals = {
    ...     "consumption elasticity": stats.uniform(0.75, 3),
    ...     "market return": stats.lognorm(0.05, 0.05),
    ...     "risk of war": stats.beta(1, 50)
    ... }

    Specify Kendall's tau (missing pairs are assumed to be independent):

    >>> tau = {
    ...     ("risk of war", "market return"): -0.5,
    ... }

    Define the joint distribution:

    >>> dist = CopulaJoint(marginals, kendall_tau=tau)

    Query ``cdf``, ``pdf`` or ``logpdf``:

    >>> dist.cdf({
    ...     "consumption elasticity": 1.5,
    ...     "market return": 1.5,
    ...     "risk of war": 0.5
    ... })
    0.24999999999998662

    Draw random samples:

    >>> sample = dist.rvs(10_000)

    Notes
    -----
    This class satisfies the same interface as other SciPy multivariate frozen distributions. There's
    no public class that they inherit from, whereas ``scipy.stats.distributions.rv_frozen`` has
    some methods that do not make sense for multidimensional distributions, such as ``ppf``.

    SciPy frozen multivariate distributions do inherit from the private
    ``scipy.stats._multivariate.multi_rv_frozen``, but that class has only the method
    ``random_state``, which is almost pointless to subclass. So I don't subclass anything.
    """

    def __init__(
        self,
        marginals: dict[Any, rv_frozen] | list[rv_frozen],
        *,
        spearman_rho: dict[tuple[Any, Any], float] | np.ndarray | float | None = None,
        kendall_tau: dict[tuple[Any, Any], float] | np.ndarray | float | None = None,
        allow_singular: bool = False,
    ):
        self.marginals = marginals
        self.family = "Gaussian"

        # For extensibility
        # So in the future, the family could be given as a string argument, e.g. "Clayton" or "StudentT"
        # The reason I haven't done this yet is that it would make the signature more complicated,
        # and require even more argument parsing code.
        # In terms of the copulas currently supported by statsmodels:
        # - For the archimedean copulas, rank correlation would need to be given as a single float.
        # - For the StudentT copula, an additional parameter of degrees of freedom would need to be given.
        CopulaClass = getattr(statsmodels.distributions.copula.api, self.family + "Copula")

        rank_corr = filter_none(spearman_rho=spearman_rho, kendall_tau=kendall_tau)

        if rank_corr.keys() == {"spearman_rho"}:
            marginals, spearman_rho = self._parse_init_args(marginals, spearman_rho)
            # This line would need to be changed if we added more copula families
            copula_corr = to_pearsons_rho(spearman=spearman_rho)
        elif rank_corr.keys() == {"kendall_tau"}:
            marginals, kendall_tau = self._parse_init_args(marginals, kendall_tau)
            # The appropriate correlation measure for the copula. For extensibility.
            copula_corr = CopulaClass()._arg_from_tau(kendall_tau)
        else:
            raise ValueError(f"Must provide exactly one of `spearman_rho` or `kendall_tau`.")

        copula = CopulaClass(copula_corr, allow_singular=allow_singular)
        self._wrapped = CopulaDistribution(copula, marginals)

    def rvs(self, size=1, random_state=None):
        array = self._wrapped.rvs(nobs=size, random_state=random_state)
        if self._idx_to_name:
            return pd.DataFrame(array, columns=self._idx_to_name)
        else:
            return array

    def pdf(self, x):
        x = self._parse_method_arg(x)
        return self._wrapped.pdf(x)

    def cdf(self, x):
        x = self._parse_method_arg(x)
        return self._wrapped.cdf(x)

    def logpdf(self, x):
        x = self._parse_method_arg(x)
        return self._wrapped.logpdf(x)

    def _parse_method_arg(self, x) -> Sequence[Real]:
        """
        Parse arguments to the ``pdf``, ``cdf``, and ``logpdf`` methods, and convert them to the
        format expected by statsmodels.
        """
        if isinstance(x, dict):
            if self._idx_to_name is None:
                raise ValueError(
                    "Cannot provide a dict argument when marginals were given as a list."
                )
            return [x[name] for name in self._name_to_idx]
        elif isinstance(x, Sequence):
            if self._idx_to_name is not None:
                raise ValueError(
                    "Cannot provide an array argument when marginals were given as a dict."
                )
            return x
        else:
            raise TypeError(f"Expected dict or Sequence, got {type(x)}.")

    def _parse_init_args(self, marginals, rank_corr) -> tuple[list[rv_frozen], np.ndarray[Real]]:
        """
        Parses arguments to the constructor, checks their validity, and converts them to the form
        expected by statsmodels.
        """
        n_dimensions = len(marginals)
        if n_dimensions < 2:
            raise ValueError("Must provide at least two marginals.")

        got_named = self._idx_to_name is not None and isinstance(rank_corr, dict)
        try:
            np.asarray(rank_corr)
            got_positional = self._idx_to_name is None
        except TypeError:
            got_positional = False
        if not xor(got_named, got_positional):
            raise ValueError("Must provide marginals and rank correlation in the same format.")

        if got_named:
            self._validate_corr_dict(rank_corr)
            rank_corr = self._to_matrix(rank_corr)
            marginals = list(marginals.values())
        if got_positional:
            rank_corr = np.asarray(rank_corr)
            if rank_corr.shape != (n_dimensions, n_dimensions):
                raise ValueError(
                    "Rank correlation matrix must be square and have the same number of rows as marginals."
                )
            if not (rank_corr == rank_corr.T).all():
                raise ValueError("Inconsistent rank correlation matrix.")

        return marginals, rank_corr

    def _to_matrix(self, corr: dict[tuple[Any, Any], float]) -> np.ndarray:
        corr_matrix = np.eye(N=len(self.marginals))
        for pair, correlation in corr.items():
            left, right = pair
            i, j = self._name_to_idx[left], self._name_to_idx[right]
            corr_matrix[i][j] = correlation
            corr_matrix[j][i] = correlation
        return corr_matrix

    def _validate_corr_dict(self, pairwise_corr: dict[tuple[Any, Any], float]):
        for pair in pairwise_corr:
            left, right = pair

            if left not in self.marginals:
                raise ValueError(f"Unknown marginal key '{left}' in correlation dict.")
            if right not in self.marginals:
                raise ValueError(f"Unknown marginal key '{right}' in correlation dict.")

            left_right = pairwise_corr[(left, right)]
            try:
                right_left = pairwise_corr[(right, left)]
            except KeyError:
                continue
            if right_left != left_right:
                raise ValueError(
                    f"Inconsistent rank correlations: {left, right}={left_right} and {right, left}={right_left}"
                )

    @property
    def _idx_to_name(self) -> list | None:
        try:
            # dicts are ordered in modern Pythons
            return list(self.marginals.keys())
        except AttributeError:
            return None

    @property
    def _name_to_idx(self) -> dict | None:
        try:
            # dicts are ordered in modern Pythons
            return {name: i for i, name in enumerate(self.marginals.keys())}
        except AttributeError:
            return None


def filter_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}
