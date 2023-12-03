def test_while(returning_executor):
    code = """
let mut x = 10
while x > 0 {
    x = x - 1
}
"""

    assert returning_executor(code, "x") == 0


def test_break(returning_executor):
    code = """
let mut x = 10
while x > 0 {
    x = x - 1
    if x == 5 {
        break
    }
}
"""

    assert returning_executor(code, "x") == 5


def test_continue(returning_executor):
    code = """
let mut x = 3
while x > 0 {
    x = x - 1
    if x == 2 {
        continue
    }
    x = x - 1
}
"""

    assert returning_executor(code, "x") == 0
