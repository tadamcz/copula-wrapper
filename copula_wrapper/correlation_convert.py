import numpy as np


def pearsons_rho(kendalls_tau=None, spearmans_rho=None):
	"""
	Bivariate Pearson's rho from bivariate rank correlations (Kendall's Tau or Spearman's Rho).

	References: https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html
	"""

	if kendalls_tau is not None and spearmans_rho is not None:
		raise ValueError("Must provide exactly one of `kendalls_tau` or `spearmans_rho`.")

	if kendalls_tau is not None:
		func = lambda kendalls_tau: np.sin(kendalls_tau * np.pi / 2)
		arg = kendalls_tau
	elif spearmans_rho is not None:
		func = lambda spearmans_rho: 2 * np.sin(spearmans_rho * np.pi / 6)
		arg = spearmans_rho
	else:
		raise ValueError("Must provide exactly one of `kendalls_tau` or `spearmans_rho`.")

	try:
		return func(arg)
	except TypeError:
		return np.vectorize(func)(arg)
