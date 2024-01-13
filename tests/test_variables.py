import pytest


def test_mutable_assignment_fails(executor):
    with pytest.raises(ValueError):
        executor("let x = 1\nx = 2")



