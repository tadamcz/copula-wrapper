import pytest

from copula_wrapper import correlation_convert


@pytest.fixture(params=['spearmans_rho', 'kendalls_tau'], ids=lambda p: f"method={p}")
def corr_method(request):
	return request.param


@pytest.fixture(params=[0, 1], ids=lambda p: f"corr={p}")
def extremes_corr(request):
	return request.param


def test_extremes(corr_method, extremes_corr):
	if corr_method == 'kendalls_tau':
		assert correlation_convert.pearsons_rho(kendalls_tau=extremes_corr) == pytest.approx(extremes_corr)

	if corr_method == 'spearmans_rho':
		assert correlation_convert.pearsons_rho(spearmans_rho=extremes_corr) == pytest.approx(extremes_corr)
