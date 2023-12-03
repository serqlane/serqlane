def test_mut_block(returning_executor):
    code = """
let mut i = 1
{
    i = i + i
}
"""

    assert returning_executor(code, "i") == 2


def test_block_expression(returning_executor):
    code = """
let x = {
    1 + 1
}
"""

    assert returning_executor(code, "x") == 2
