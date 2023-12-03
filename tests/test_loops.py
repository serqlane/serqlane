from collections.abc import Callable
from typing import Any

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


def test_while(checking_executor):
    checking_executor("""
let mut x = 10
while x > 0 {
    x = x - 1
}
""", "x", 0)

def test_break(checking_executor):
    checking_executor("""
let mut x = 10
while x > 0 {
    x = x - 1
    if x == 5 {
        break
    }
}
""", "x", 5)

def test_continue(checking_executor):
    checking_executor("""
let mut x = 3
while x > 0 {
    x = x - 1
    if x == 2 {
        continue
    }
    x = x - 1
}
""", "x", 0)
