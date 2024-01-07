import pytest


# [({module_name: code}, expected)]
import_tests = [
    (
        {"main": """
import other

fn a() -> int64 {
    other.change_x(20)
}

dbg(a())
""", "other": """
pub fn change_x(value: int64) -> int64 {
    let mut x = 1
    x = value
    return x
}
"""},
        20
    ),
    ({"main": """
from math import *
dbg(add(20, 10))
""", "math": """
pub fn add(a: int64, b: int64) -> int64 {
    a + b
}
"""}, 30),
    ({"main": """
from math import *
dbg(math.add(20, 20))
""", "math": """
pub fn add(a: int64, b: int64) -> int64 {
    a + b
}
"""}, 40),
    ({"main": """
from math import [add]
dbg(add(25, 25))
""", "math": """
pub fn add(a: int64, b: int64) -> int64 {
    a + b
}
"""}, 50),
    ({"main": """
from math import [add, sub]
dbg(add(30, sub(40, 10)))
""", "math": """
pub fn add(a: int64, b: int64) -> int64 {
    a + b
}

pub fn sub(a: int64, b: int64) -> int64 {
    a - b
}
"""}, 60),
]


@pytest.mark.parametrize("modules,expected", import_tests)
def test_imports_passing(multimodule_capture_first_debug, modules, expected):
    assert multimodule_capture_first_debug(modules) == expected


# for some reason this is allowed
def test_no_op_import(multimodule_executor):
    multimodule_executor({"main": "from other import []", "other": ""})
