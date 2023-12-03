from typing import Any, Callable
import pytest

from serqlane.astcompiler import ModuleGraph
from serqlane.vm import SerqVM


@pytest.fixture
def checking_executor() -> Callable[[str, str, Any], None]:
    def execute_with_check(code: str, variable: str, value: Any):
        graph = ModuleGraph()
        vm = SerqVM()
        module = graph.load("<string>", code)
        vm.execute_module(module)

        assert vm.get_stack_value_by_name(variable) == value

    return execute_with_check


def test_mut_block(checking_executor):
    checking_executor(
        """
let mut i = 1
{
    i = i + i
}
""", "i", 2
    )


def test_block_expression(checking_executor):
    checking_executor("""
let x = {
    1 + 1
}
""", "x", 2)

