import pytest


# code, variable, expected
return_tests = [
    (
        """
fn add(a: int, b: int): int {
    return a + b
}

let x = add(1, 1)
""",
        "x",
        2,
    ),
    (
        """
fn add(a: int, b: int): int {
    a + b
}

let x = add(1, 1)
""",
        "x",
        2,
    ),
]


def test_variable_shadowing(executor):
    executor(
        """
let x = 1

fn foo(x: int): int {
    return x
}
"""
    )


@pytest.mark.parametrize("code,variable,expected", return_tests)
def test_return(returning_executor, code, variable, expected):
    assert returning_executor(code, variable) == expected


def test_recusive_function(returning_executor):
    code = """
fn add_one(x: int): int {
    if x == 1 {
        // should give us 4
        return add_one(x + 1) + 1
    }

    x + 1
}

let w = add_one(1)
"""

    assert returning_executor(code, "w") == 4
