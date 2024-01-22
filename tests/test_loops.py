from typing import Any

import pytest


def test_while(capture_first_debug):
    code = """
let mut x = 10
while x > 0 {
    x = x - 1
}
dbg(x)
"""

    assert capture_first_debug(code) == 0


def test_break(capture_first_debug):
    code = """
let mut x = 10
while x > 0 {
    x = x - 1
    if x == 5 {
        break
    }
}
dbg(x)
"""

    assert capture_first_debug(code) == 5


def test_continue(capture_first_debug):
    code = """
let mut x = 3
while x > 0 {
    x = x - 1
    if x == 2 {
        continue
    }
    x = x - 1
}
dbg(x)
"""

    assert capture_first_debug(code) == 0


# code, expected
for_loop_tests: list[tuple[str, list[Any]]] = [
    ("""
for i in 0 .. 10 {
    dbg(i)
}
""", [0,1,2,3,4,5,6,7,8,9]),("""
let i = 100

for i in 0 .. 6 {
    dbg(i)
}
""", [0,1,2,3,4,5]),("""
for i in 0 .. 5 {
    if i == 1 {
        continue
    }
    dbg(i)
}
""", [0,2,3,4]),("""
for i in 0 .. 5 {
    if i == 3 {
        break
    }
    dbg(i)
}
""", [0,1,2]),
]


@pytest.mark.parametrize("code,expected", for_loop_tests)
def test_for_loop(capture_with_debug_hook, code: str, expected: Any):
    def debug_hook_outer(capture: list[Any]):
        def debug_hook(value: Any):
            capture.append(value)

        return debug_hook

    assert capture_with_debug_hook(debug_hook_outer, code) == expected
