from typing import Any

import pytest


# code,expected
if_tests: list[tuple[str, Any]] = [
    ("""
if true {
    dbg(1)
}
""", 1),
    ("""
if false {
    dbg(1)
} else {
    dbg(2)
}
""", 2),
]


@pytest.mark.parametrize("code,expected", if_tests)
def test_if(capture_first_debug, code: str, expected: Any):
    assert capture_first_debug(code) == expected

