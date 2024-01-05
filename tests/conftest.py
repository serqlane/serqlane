from collections.abc import Callable
from typing import Any
import pytest

from serqlane.vm import SerqVM
from serqlane.astcompiler import ModuleGraph


type ModuleMap = dict[str, str]


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
def multimodule_executor() -> Callable[[ModuleMap], SerqVM]:
    def execute(modules: ModuleMap, *, debug_hook = None):
        graph = ModuleGraph()
        vm = SerqVM(debug_hook=debug_hook)

        other_modules = []
        for other_module_name, other_module_code in modules.items():
            if other_module_name == "main":
                continue

            other_modules.append(graph.load(other_module_name, other_module_code))

        main_module = graph.load("main", file_contents=modules["main"])
        vm.execute_module(main_module)
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


@pytest.fixture
def multimodule_capture_first_debug(multimodule_executor) -> Callable[[ModuleMap], Any]:
    def execute_and_capture(modules: ModuleMap) -> Any:
        class MISSING: ...

        captured = MISSING()

        def _debug_hook(value: Any):
            nonlocal captured
            if not isinstance(captured, MISSING):
                raise RuntimeError("dbg called twice")
            
            captured = value

        multimodule_executor(modules, debug_hook=_debug_hook)

        if isinstance(captured, MISSING):
            raise RuntimeError("dbg not called")
        
        return captured

    return execute_and_capture

