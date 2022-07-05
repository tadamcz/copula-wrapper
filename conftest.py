import os

import numpy as np
import pytest

import tests.seeds as seeds


def seed_idfn(fixture_value):
	return f"seed={fixture_value}"


n_random_seeds = int(os.environ.get('N_RAND_SEED', 10))


@pytest.fixture(autouse=True, params=seeds.RANDOM_SEEDS[:n_random_seeds], ids=seed_idfn, scope='session')
def random_seed(request):
	"""
	autouse:
	this fixture will be used by every test, even if not explicitly requested.

	params:
	this fixture will be run once for each element in params


	scope:
	setting scope to 'session' is the easiest way to control the ordering,
	so that all tests are run for RANDOM_SEEDS[0], then all tests for RANDOM_SEEDS[1], etc.
	"""
	np.random.seed(request.param)
