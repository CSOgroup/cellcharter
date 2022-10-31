import pytest

import cellcharter


def test_package_has_version():
    cellcharter.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.
