import pytest


def test_invalid_cohersion(executor):
    # TODO: this should probably raise a better exception
    with pytest.raises(AssertionError):
        executor('let x: int = "abc"')


def test_floating_function_return_type(executor):
    with pytest.raises(AssertionError):
        executor(
            """
fn add(a: int, b: int) -> string {
    a + b
}

let x = add(1, 1)
"""
        )


def test_empty_literal(capture_first_debug):
    assert capture_first_debug("dbg(\"\")") == ""


BAD_STRUCT_TYPE_INFERENCE = [
"""
struct A {
    x: s32
}

let w: f32 = A().x
""",
"""
struct A {
    x: s32
}

let w = A()
let q = w.x + "def"
"""
]


@pytest.mark.parametrize("code", BAD_STRUCT_TYPE_INFERENCE)
def test_bad_struct_type_inference(executor, code):
    with pytest.raises(ValueError):
        executor(code)


GOOD_STRUCT_TYPE_INFERENCE = [
"""
struct A {
    x: s32
}

let w: s32 = A().x
""",
"""
struct A {
    x: s32
}

let w = A()
let q = w.x + 2
"""
]


@pytest.mark.parametrize("code", GOOD_STRUCT_TYPE_INFERENCE)
def test_good_struct_type_inference(executor, code):
    executor(code)
