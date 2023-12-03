from collections.abc import Callable
from typing import Any

import pytest

from serqlane.astcompiler import ModuleGraph
from serqlane.vm import SerqVM


@pytest.fixture
def executor() -> Callable[[str], SerqVM]:
    def execute(code: str):
        graph = ModuleGraph()
        vm = SerqVM()
        module = graph.load("<string>", code)
        vm.execute_module(module)

        return vm

    return execute


@pytest.fixture
def checking_executor() -> Callable[[str, str, Any], None]:
    def execute_with_check(code: str, variable: str, value: Any):
        graph = ModuleGraph()
        vm = SerqVM()
        module = graph.load("<string>", code)
        vm.execute_module(module)

        assert vm.get_stack_value_by_name(variable) == value

    return execute_with_check


def test_variable_shadowing(executor):
    executor("""
let x = 1

fn foo(x: int): int {
    return x
}
""")


def test_return(checking_executor):
    checking_executor("""
fn add(a: int, b: int): int {
    return a + b
}

let x = add(1, 1)
""", "x", 2)
