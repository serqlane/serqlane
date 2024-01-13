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
    ("dbg(1.1 + 1.1)", 2.2),
    ("dbg(4 and 2)", 0),
    ("dbg(true or false)", True),
    ("dbg(1 != 2)", True),
    ("dbg(1 < 2)", True),
    ("dbg(2 <= 2)", True),
    ("dbg(2 > 3)", False),
    ("dbg(2 >= 2)", True),
]

variable_arith_tests = [
    ("""
let x = 1
let y = x + 2
dbg(y)
""",3),
    ("""
let x = 1
let y = x + x
dbg(y)
""",2),
    ("""
let x = 1
let y = x * x
dbg(y)
""",1),
    ("""
let x = 8
let y = x % x
dbg(y)
""",0),
    ("""
let x = true
let y = x != x
dbg(y)
""",False),
    ("""
let x = 200
let y = x < 300
dbg(y)
""",True),
    ("""
let x = 300
let y = x <= 300
dbg(y)
""",True),
    ("""
let x = 300
let y = x >= 300
dbg(y)
""",True),
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
    """
let x = 1
let y = x + 1.1
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
