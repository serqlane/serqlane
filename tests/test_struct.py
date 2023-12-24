def test_chained(capture_first_debug):
    assert capture_first_debug("""
struct A {
    x: int
}

struct B {
    a: A
}

let mut b = B()
b.a.x = 100
dbg(b.a.x)
""") == 100
