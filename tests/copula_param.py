import pytest

from copula_wrapper import correlation_convert


@pytest.fixture(params=['spearman', 'kendall'], ids=lambda p: f"method={p}")
def corr_method(request):
	return request.param


@pytest.fixture(params=[0, 1], ids=lambda p: f"corr={p}")
def extremes_corr(request):
	return request.param


def test_extremes(corr_method, extremes_corr):
	if corr_method == 'kendall':
		assert correlation_convert.pearsons_rho(kendall=extremes_corr) == pytest.approx(extremes_corr)

	if corr_method == 'spearman':
		assert correlation_convert.pearsons_rho(spearman=extremes_corr) == pytest.approx(extremes_corr)
