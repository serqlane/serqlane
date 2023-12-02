from collections.abc import Callable
from typing import Any
import pytest

from serqlane.vm import SerqVM
from serqlane.astcompiler import ModuleGraph


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


def test_literal_arith(checking_executor: Callable[[str, str, Any], None]):
    checking_executor("let x = 1 + 1;", "x", 2)
    checking_executor("let x = 1 - 1;", "x", 0)
    checking_executor("let x = 1 * 2;", "x", 2)
    checking_executor("let x = 1 / 1;", "x", 1)

    checking_executor("let x = 1 + 1 + 1;", "x", 3)
    checking_executor("let x = 10 / (3 + 2);", "x", 2)
    checking_executor("let x = (10 / (3 + 2));", "x", 2)


def test_variable_arith(checking_executor: Callable[[str, str, Any], None]):
    checking_executor(
        """
let x = 1;
let y = x + 2;
""",
        "y",
        3,
    )

    checking_executor(
        """
let x = 1;
let y = x + x;
""",
        "y",
        2,
    )
