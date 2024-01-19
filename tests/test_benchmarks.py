from serqlane.astcompiler import ModuleGraph
from serqlane.vm import SerqVM

try:
    import pytest_benchmark
except ImportError:
    pass
else:
    def test_benchmark_parser(benchmark):
        testcode = "let mut x = 1" + "\nx = x + 1" * 10_000
        benchmark(lambda p, f: ModuleGraph().load(p, f), p="test", f=testcode)


    def test_vm(benchmark):
        testcode = "let mut x = 1" + "\nx = x + 1" * 10_000
        graph = ModuleGraph()
        module = graph.load("test", testcode)
        vm = SerqVM()

        benchmark(vm.execute_module, module)
