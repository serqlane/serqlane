import pytest


# code, variable, expected
return_tests = [
    (
        """
fn add(a: int, b: int) -> int {
    return a + b
}

let x = add(1, 1)
dbg(x)
""",
        2,
    ),
    (
        """
fn add(a: int, b: int) -> int {
    a + b
}

let x = add(1, 1)
dbg(x)
""",
        2,
    ),
]


def test_variable_shadowing(capture_first_debug):
    code = """
let x = 1

fn foo(x: int) -> int {
    return x
}

foo(20)
dbg(x)
"""

    assert capture_first_debug(code) == 1


@pytest.mark.parametrize("code,expected", return_tests)
def test_return(capture_first_debug, code, expected):
    assert capture_first_debug(code) == expected


def test_recusive_function(capture_first_debug):
    code = """
fn add_one(x: int) -> int {
    if x == 1 {
        // should give us 4
        return add_one(x + 1) + 1
    }

    x + 1
}

let w = add_one(1)
dbg(w)
"""

    assert capture_first_debug(code) == 4


def test_function_empty_args(executor):
    executor("""
fn abc() {
    return
}
""")


def test_function_call_symbol(executor):
    with pytest.raises(AssertionError):
        executor("""
true()
""")
