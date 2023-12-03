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
