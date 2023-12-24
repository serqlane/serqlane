import pytest


def test_chained(capture_first_debug):
    assert (
        capture_first_debug(
            """
struct A {
    x: int
}

struct B {
    a: A
}

let mut b = B()
b.a.x = 100
dbg(b.a.x)
"""
        )
        == 100
    )


def test_multi_instance(capture_first_debug):
    assert (
        capture_first_debug(
            """
struct A {
    x: int
}

let mut a = A()
let mut b = A()
let mut c = A()
a.x = 1
b.x = 2
c.x = 3
dbg(b.x)
"""
        )
        == 2
    )


def test_mutability(executor):
    with pytest.raises(ValueError):
        executor(
            """
struct A {
    x: int
}

struct B {
    a: A
}

let b = B()
b.a.x = 100
dbg(b.a.x)
"""
)


def test_function_dot_access(capture_first_debug):
    assert (
        capture_first_debug(
            """
struct A {
    x: int
}

fn give_a() -> A {
    let mut result = A()
    result.x = 20
    result
}

dbg(give_a().x)
"""
        )
        == 20
    )
