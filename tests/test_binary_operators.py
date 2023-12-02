from collections.abc import Callable
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


def test_literal_arith(executor: Callable[[str], SerqVM]):
    vm = executor("let x = 1 + 1;")
    assert vm.get_stack_value_by_name("x") == 2

    vm = executor("let x = 1 - 1;")
    assert vm.get_stack_value_by_name("x") == 0

    vm = executor("let x = 1 / 1;")
    assert vm.get_stack_value_by_name("x") == 1
