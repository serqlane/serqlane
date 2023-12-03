from collections.abc import Callable

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


def test_invalid_cohersion(executor):
    # TODO: this should probably raise a better exception
    with pytest.raises(AssertionError):
        executor("let x: int = \"abc\"")

def test_floating_function_return_type(executor):
    with pytest.raises(AssertionError):
        executor("""
fn add(a: int, b: int): string {
    a + b
}

let x = add(1, 1)
""")
