import pytest

# code, variable, expected
literal_arith_tests = [
    ("let x = 1 + 1", "x", 2),
    ("let x = 1 - 1", "x", 0),
    ("let x = 1 * 2", "x", 2),
    ("let x = 1 / 1", "x", 1),
    ("let x = 1 + 1 + 1", "x", 3),
    ("let x = 10 / (3 + 2)", "x", 2),
    ("let x = (10 / (3 + 2))", "x", 2)
]

variable_arith_tests = [
    ("""
let x = 1
let y = x + 2
""", "y", 3),
    ("""
let x = 1
let y = x + x
""",
    "y",
    2)
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
"""
]

@pytest.mark.parametrize("code,variable,expected", literal_arith_tests)
def test_literal_arith(returning_executor, code, variable, expected):
    assert returning_executor(code, variable) == expected
    

@pytest.mark.parametrize("code,variable,expected", variable_arith_tests)
def test_variable_arith(returning_executor, code, variable, expected):
    assert returning_executor(code, variable) == expected


@pytest.mark.parametrize("code", type_inference_expected_failure_tests)
def test_type_inference_expected_failures(executor, code):
    with pytest.raises(ValueError):
        executor(code)

def test_type_inference(returning_executor):
    code = """
let x = 100
let y = true
let z = x > 0 and (x == 100) or false
"""

    assert returning_executor(code, "z") == True
