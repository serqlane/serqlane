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


def test_invalid_cohersion(executor: Callable[[str], SerqVM]):
    # TODO: this should probably raise a better exception
    with pytest.raises(AssertionError):
        executor("let x: int = \"abc\";")
