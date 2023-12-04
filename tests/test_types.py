import pytest


def test_invalid_cohersion(executor):
    # TODO: this should probably raise a better exception
    with pytest.raises(AssertionError):
        executor('let x: int = "abc"')


def test_floating_function_return_type(executor):
    with pytest.raises(AssertionError):
        executor(
            """
fn add(a: int, b: int) -> string {
    a + b
}

let x = add(1, 1)
"""
        )


def test_empty_literal(capture_first_debug):
    assert capture_first_debug("dbg(\"\")") == ""

