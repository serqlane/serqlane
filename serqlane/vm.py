import operator
from typing import Any

from serqlane.astcompiler import *


class SerqVM:
    def __init__(self) -> None:
        self.stack: list[dict[Symbol, Any]] = []

    # useful for tests
    def get_stack_value_by_name(self, name: str):
        frame = self.stack[-1]
        for symbol, value in frame.items():
            if symbol.name == name:
                return value
            
        raise ValueError(f"{name} not found in stack frame")

    def eval_binary_expression(self, expression: NodeBinaryExpr) -> Any:
        match expression:
            case NodePlusExpr():
                operation = operator.add
            case NodeMinusExpression():
                operation = operator.sub
            case NodeMulExpression():
                operation = operator.mul
            case NodeDivExpression():
                assert expression.type is not None

                # TODO: should this be handled like this?
                if expression.type.kind in int_types:
                    operation = operator.floordiv
                else:
                    if expression.type.kind is TypeKind.literal_int:
                        raise RuntimeError(f"Got literal int in {expression}")

                    operation = operator.truediv
                
                print(f"{operation=} {expression=} {expression.type.kind=}")

            case NodeModExpression():
                operation = operator.mod
            case NodeAndExpression():
                operation = operator.and_
            case NodeOrExpression():
                operation = operator.or_
            case NodeEqualsExpression():
                operation = operator.eq
            case NodeNotEqualsExpression():
                operation = operator.ne
            case NodeLessExpression():
                operation = operator.lt
            case NodeLessEqualsExpression():
                operation = operator.le
            case NodeGreaterExpression():
                operation = operator.gt
            case NodeGreaterEqualsExpression():
                operation = operator.ge
            case NodeDotExpr():
                raise NotImplementedError()
            case NodeBinaryExpr():
                raise RuntimeError("uninstatiated binary expression")

        return operation(self.eval(expression.lhs), self.eval(expression.rhs))

    def eval(self, expression: Node) -> Any:
        match expression:
            case NodeLiteral():
                return expression.value

            case NodeBinaryExpr():
                return self.eval_binary_expression(expression)

            case Symbol():
                return self.stack[-1][expression]

            case NodeGrouped():
                return self.eval(expression.inner)

            case _:
                raise NotImplementedError(f"{expression=}")

    def enter_scope(self):
        self.stack.append({})

    def exit_scope(self):
        self.stack.pop()

    def push_scope(self, symbol: Symbol, value: Any):
        self.stack[-1][symbol] = value

    def execute_module(self, module: Module):
        start = module.ast

        assert start is not None and isinstance(start, NodeStmtList)

        self.enter_scope()

        for child in start.children:
            match child:
                case NodeLet():
                    self.push_scope(child.sym, self.eval(child.expr))
            print(f"{self.stack=}")


if __name__ == "__main__":
    code = """
let x: int = "abc";
"""

    graph = ModuleGraph()
    module = graph.load("<string>", code)
    print(module.ast.render())

    vm = SerqVM()
    vm.execute_module(module)

