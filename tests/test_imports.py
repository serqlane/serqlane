import pytest


# [({module_name: code}, expected)]
import_tests = [
    (
        {"main": """
import other

fn a() -> int64 {
    other.change_x(20)
}

dbg(a())
""", "other": """
pub fn change_x(value: int64) -> int64 {
    let mut x = 1
    x = value
    return x
}
"""},
        20
    )
]


@pytest.mark.parametrize("modules,expected", import_tests)
def test_imports_passing(multimodule_capture_first_debug, modules, expected):
    assert multimodule_capture_first_debug(modules) == expected























