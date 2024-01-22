import pytest

# code, expected
literal_arith_tests = [
    ("dbg(1 + 1)", 2),
    ("dbg(1 - 1)", 0),
    ("dbg(1 * 2)", 2),
    ("dbg(1 / 1)", 1),
    ("dbg(1 + 1 + 1)", 3),
    ("dbg(10 / (3 + 2))", 2),
    ("dbg((10 / (3 + 2)))", 2),
    ("dbg(10 % 3)", 1),
]

variable_arith_tests = [
    (
        """
let x = 1
let y = x + 2
dbg(y)
""",
        3,
    ),
    (
        """
let x = 1
let y = x + x
dbg(y)
""",
        2,
    ),
    (
        """
const a = 10
let b: int32 = a
let c: int64 = a
dbg(c + a)
""",
        20
    )
]

# code
type_inference_expected_failure_tests = [
    """
let x = "abc"
let y = x + 1
""",
    """
let x: int32 = 10
let y: int64 = 20
let y: int64 = (x + x) - y
""",
    """
let x = 1 == true
""",
]



@pytest.mark.parametrize("code,expected", literal_arith_tests)
def test_literal_arith(capture_first_debug, code, expected):
    assert capture_first_debug(code) == expected


@pytest.mark.parametrize("code,expected", variable_arith_tests)
def test_variable_arith(capture_first_debug, code, expected):
    assert capture_first_debug(code) == expected


@pytest.mark.parametrize("code", type_inference_expected_failure_tests)
def test_type_inference_expected_failures(executor, code):
    with pytest.raises(ValueError):
        executor(code)


def test_type_inference(capture_first_debug):
    code = """
let x = 100
let y = true
let z = x > 0 and (x == 100) or false
dbg(z)
"""

    assert capture_first_debug(code) == True
