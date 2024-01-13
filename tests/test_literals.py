from typing import Any
import pytest


def test_empty_literal(capture_first_debug):
    assert capture_first_debug("dbg(\"\")") == ""


def test_quote_escape(capture_first_debug):
    assert capture_first_debug('dbg("\\"")') == '"'


# code, expected
render_tests: list[tuple[str, str]] = [
("let x = 1 + 1", "let x: int64 = 2"),
("""
let x = 1
let y = 1 + 1 + x
""","""
let x: int64 = 1
let y: int64 = (2 + x)
"""),
("let x = true and false", "let x: bool = false"),
("let x = 1 == 1", "let x: bool = true"),
("let x = 1 or 2", "let x: int64 = 3"),
]


@pytest.mark.parametrize("code,expected", render_tests)
def test_folding(renderer, code: str, expected: str):
    assert renderer(code).replace("\n", "") == expected.replace("\n", "")


# code, expected
basic_literal_tests: list[tuple[str, Any]] = [
    ("dbg(1)", 1),
    ("dbg(1.2)", 1.2),
    ("dbg(true)", True),
    ("dbg(false)", False),
    ('dbg("abc")', "abc"),
]


@pytest.mark.parametrize("code,expected", basic_literal_tests)
def test_basic_literal(capture_first_debug, code: str, expected: str):
    assert capture_first_debug(code) == expected
