import pytest

from serqlane.astcompiler import SerqInternalError


# code, variable, expected
return_tests = [
    (
        """
alias int = int64

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
alias int = int64
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
alias int = int64

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
alias int = int64

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
abc()
""")


def test_function_call_symbol(executor):
    with pytest.raises(ValueError):
        executor("""
true()
""")


# code, expected
overload_tests = [
("""
fn abc(a: int64) -> int64 {
    a
}

fn abc(a: string) -> string {
    a
}

dbg(abc(1))
""", 1),
(
"""
alias int = int64

fn abc() -> int {
    1
}

fn abc(a: int) -> int {
    a
}

dbg(abc())
""", 1),
(
"""
alias int = int64

fn abc(a: int) -> int {
    a
}

fn abc(a: int, b: string) -> int {
    a
}

dbg(abc(1))
""", 1)
]


@pytest.mark.parametrize("code,expected", overload_tests)
def test_overloads(capture_first_debug, code, expected):
    assert capture_first_debug(code) == expected


# code, expected
forward_calls_tests_passing = [
("""
fn a() {
 b()
}

fn b() {
 dbg(1)
}

a()
""", 1), (
"""
fn foo() -> int32 {
    bar(10)
}

fn bar(a: int32) -> int32 {
    a
}

dbg(foo())
""", 10
)
]


# code
forward_calls_tests_failing = [
"""
fn a() {}

b()

fn b() {}
"""
]


@pytest.mark.parametrize("code,expected", forward_calls_tests_passing)
def test_forward_calls(capture_first_debug, code, expected):
    assert capture_first_debug(code) == expected


@pytest.mark.parametrize("code", forward_calls_tests_failing)
def test_forward_calls_failing(executor, code):
    with pytest.raises(ValueError):
        executor(code)


def test_redefinition_fails(executor):
    with pytest.raises(ValueError):
        executor("fn a() {}\nfn a() {}")
