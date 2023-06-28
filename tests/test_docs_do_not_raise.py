import pytest
from mkcodes import github_codeblocks

from copula_wrapper import PROJECT_ROOT


@pytest.fixture(params=github_codeblocks(PROJECT_ROOT / "README.md", safe=True)["py"])
def block(request):
    return request.param


def test_docs_do_not_raise(block):
    exec(block)
