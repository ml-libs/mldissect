import pytest


@pytest.fixture(scope='session')
def seed():
    return 0


pytest_plugins = []
