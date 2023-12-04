def test_unary_minus(capture_first_debug):
    code = """
dbg(-10 + 10)
"""
    assert capture_first_debug(code) == 0


def test_unary_plus(capture_first_debug):
    code = """
dbg(+10 + 10)
"""
    assert capture_first_debug(code) == 20