import pytest


def test_pub_const(multimodule_capture_first_debug):
    assert multimodule_capture_first_debug(
        {"main": """
import other
dbg(other.a)
""", "other": """
pub const a = 100
"""}
    ) == 100


def test_non_pub_const_failure(multimodule_executor):
    with pytest.raises(ValueError):
        multimodule_executor({"main": "import other\ndbg(other.a)", "other": "const a = 100"})