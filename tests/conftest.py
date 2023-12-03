from collections.abc import Callable
from typing import Any
import pytest

from serqlane.vm import SerqVM
from serqlane.astcompiler import ModuleGraph


@pytest.fixture
def executor() -> Callable[[str], SerqVM]:
    def execute(code: str, *, debug_hook = None):
        graph = ModuleGraph()
        vm = SerqVM(debug_hook=debug_hook)
        module = graph.load("<string>", code)
        vm.execute_module(module)
        return vm

    return execute


@pytest.fixture
def capture_first_debug(executor) -> Callable[[str], Any]:
    def execute_and_capture(code: str) -> Any:
        class MISSING: ...

        captured = MISSING()

        def _debug_hook(value: Any):
            nonlocal captured
            if not isinstance(captured, MISSING):
                raise RuntimeError("dbg called twice")
            
            captured = value

        executor(code, debug_hook=_debug_hook)

        if isinstance(captured, MISSING):
            raise RuntimeError("dbg not called")
        
        return captured

    return execute_and_capture
