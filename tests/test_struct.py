import pytest


def test_chained(capture_first_debug):
    assert (
        capture_first_debug(
            """
struct A {
    x: int64
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
    x: int64
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
    x: int64
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
            alias int = int64
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


# [{name: code}]
valid_publics: list[dict[str, str]] = [
    {"main": """
import other

let abc = other.make_abc()

let x = abc.x
""", "other": """
struct Abc {
    pub x: int64
}

pub fn make_abc() -> Abc {Abc()}
"""},{"main": """
import other
      
let abc = other.Abc()
let x = abc.x
""", "other": """
pub struct Abc {
    pub x: int64
}
"""}
]


@pytest.mark.parametrize("modules", valid_publics)
def test_valid_publics(multimodule_executor, modules: dict[str, str]):
    multimodule_executor(modules)


# [({name: code}, raises)]
invalid_publics: list[tuple[dict[str, str], type[Exception]]] = [
    ({"main": """
import other

let abc = other.make_abc()

let x = abc.x
""", "other": """
struct Abc {
    x: int64
}

pub fn make_abc() -> Abc {Abc()}
"""}, AssertionError)
]


@pytest.mark.parametrize("modules,raises", invalid_publics)
def test_invalid_publics(multimodule_executor, modules: dict[str, str], raises: type[Exception]):
    with pytest.raises(raises):
        multimodule_executor(modules)
