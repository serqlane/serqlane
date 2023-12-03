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
def returning_executor(executor) -> Callable[[str, str], Any]:
    def execute_then_return(code: str, variable: str) -> Any:
        vm = executor(code)
        return vm.get_stack_value_by_name(variable)

    return execute_then_return
