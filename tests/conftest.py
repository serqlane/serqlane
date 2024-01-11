from collections.abc import Callable
from typing import Any
import pytest

from serqlane.vm import SerqVM
from serqlane.astcompiler import ModuleGraph, Module


type ModuleMap = dict[str, str]


@pytest.fixture(scope="session")
def module_graph() -> ModuleGraph:
    graph = ModuleGraph()
    graph.request_module("magics")
    return graph

def clear_graph(graph: ModuleGraph):
    graph.modules = {"magics": graph.modules["magics"]}


# TODO: multi-module version
@pytest.fixture
def serq_module(module_graph: ModuleGraph) -> Callable[[str], Module]:
    def execute(code: str):
        clear_graph(module_graph)
        return module_graph.load("<string>", code)
    
    return execute


@pytest.fixture
def renderer(serq_module) -> Callable[[str], str]:
    def execute(code: str):
        # TODO: don't do this removeprefix
        return serq_module(code).ast.render().removeprefix("from magics import *\n")

    return execute


@pytest.fixture
def executor(serq_module) -> Callable[[str], SerqVM]:
    def execute(code: str, *, debug_hook = None):
        module = serq_module(code)
        vm = SerqVM(debug_hook=debug_hook)
        vm.execute_module(module)
        return vm

    return execute


@pytest.fixture
def multimodule_executor(module_graph: ModuleGraph) -> Callable[[ModuleMap], SerqVM]:
    def execute(modules: ModuleMap, *, debug_hook = None):
        clear_graph(module_graph)
        vm = SerqVM(debug_hook=debug_hook)

        other_modules = []
        for other_module_name, other_module_code in modules.items():
            if other_module_name == "main":
                continue

            other_modules.append(module_graph.load(other_module_name, other_module_code))

        main_module = module_graph.load("main", file_contents=modules["main"])
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



