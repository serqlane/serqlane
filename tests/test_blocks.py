def test_mut_block(capture_first_debug):
    code = """
let mut i = 1
{
    i = i + i
}
dbg(i)
"""

    assert capture_first_debug(code) == 2


def test_block_expression(capture_first_debug):
    code = """
let x = {
    1 + 1
}
dbg(x)
"""

    assert capture_first_debug(code) == 2
