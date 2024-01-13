import sys


from serqlane.astcompiler import ModuleGraph
from serqlane.vm import SerqVM


def main():
    filename = sys.argv[1]

    #with open(filename) as fp:
    #    code = fp.read()

    graph = ModuleGraph()
    module = graph.request_module(filename)

    vm = SerqVM()
    vm.execute_module(module)


if __name__ == "__main__":
    main()
