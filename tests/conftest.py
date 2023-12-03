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
def checking_executor(executor) -> Callable[[str, str, Any], None]:
    def execute_with_check(code: str, variable: str, value: Any):
        vm = executor(code)
        assert vm.get_stack_value_by_name(variable) == value

    return execute_with_check

