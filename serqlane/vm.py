import ctypes
import logging
import operator
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


from serqlane.astcompiler import *


class SerqVMError(Exception):
    ...


class ContinueError(SerqVMError):
    ...


class BreakError(SerqVMError):
    ...


class ReturnError(SerqVMError):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class DiagnosticError(SerqVMError):
    ...


class StructNotDefined(DiagnosticError):
    ...


class Register:
    def __init__(self, name: str, *, use_once: bool = True) -> None:
        self.name = name
        self.use_once = use_once

        self._value: Any = None
        self._set: bool = False

    def __repr__(self) -> str:
        return f"<Register name={self.name}, set={self._set}, value={self._value}>"

    def is_set(self) -> bool:
        return self._set

    def get_value(self) -> Any:
        if self._set:
            if self.use_once:
                self._set = False

            return self._value
        else:
            raise ValueError(f"called get_value on unset register {self.name}")

    def set_value(self, value: Any, *, ignore_prev: bool = True):
        if not ignore_prev and self._set:
            raise ValueError(f"called set_value on already set register {self.name}")

        logger.debug(f"setting value of register {self.name} to {value}")

        self._value = value
        self._set = True


class Unit:
    ...


SERQ_TO_C_TYPE = {
    TypeKind.int32: ctypes.c_int32,
    TypeKind.float32: ctypes.c_float,
    TypeKind.int64: ctypes.c_int64,
    TypeKind.float64: ctypes.c_double,
}


class SerqVM:
    def __init__(self, *, debug_hook: Callable[[Any], None] | None = None) -> None:
        self.debug_hook = debug_hook
        self.stack: list[dict[Symbol, Any]] = []

    def construct_serq_struct(self, struct: NodeStructDefinition):
        name = struct.sym.name

        fields: list[tuple[str, ctypes._SimpleCData]] = []
        for field in struct.fields:
            field_name = field.sym.qualified_name()
            match field.type.kind:
                case TypeKind.type:
                    try:
                        type = self.get_value_on_stack(field.type.sym)
                    except KeyError:
                        raise StructNotDefined(f"{field.type.sym.name} not defined")
                case _:
                    try:
                        # TODO: lookup other structs
                        type = SERQ_TO_C_TYPE[field.type.kind]
                    except KeyError:
                        raise NotImplementedError(f"field of type {field.type.kind} not implemented")

            fields.append((field_name, type))

        class SerqStruct(ctypes.Structure):
            _fields_ = fields

        SerqStruct.__name__ = name

        return SerqStruct

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
            case NodeDotAccess():
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

            case NodeSymbol():
                return self.get_value_on_stack(expression.symbol)

            case NodeGrouped():
                return self.eval(expression.inner)

            case NodeFnCall():
                if expression.callee.symbol.name == "dbg":  # type: ignore (olaf code)
                    val = self.eval(expression.args[0])
                    if self.debug_hook is None:
                        print(f"DBG: {val}")
                    else:
                        self.debug_hook(val)
                    return Unit()
                
                # TODO: handle passing args to constructor
                elif isinstance(expression.callee.symbol.definition_node, NodeStructDefinition):  # type: ignore
                    assert isinstance(expression.callee, NodeSymbol)
                    struct = self.get_value_on_stack(expression.callee.symbol)
                    # this is a ctypes.Structure type
                    return struct()

                else:
                    stack = self.stack.copy()
                    self.enter_scope()

                    # TODO: Change these lines once function pointers exist
                    assert isinstance(expression.callee, NodeSymbol)
                    # TODO: make symbol generic for definition node
                    fn_def: NodeFnDefinition = expression.callee.symbol.definition_node  # type: ignore (olaf code)
                    for i in range(0, len(expression.args)):
                        val = self.eval(expression.args[i])
                        sym = fn_def.params.args[i][0].symbol
                        self.push_value_on_stack(sym, val)

                    try:
                        self.execute_node(fn_def.body)
                    except ReturnError as e:
                        return_value = e.value
                    else:
                        raise RuntimeError(
                            f"None return from function {expression.callee.symbol}"
                        )

                    self.exit_scope()  # exit function scope

                    self.stack = stack
                    return return_value

            # TODO: add NodeBlockExpression
            # NOTE: this is actually a block expression
            case NodeBlockStmt():
                stack = self.stack.copy()
                self.enter_scope()

                return_value = self.execute_node(expression)

                if return_value is None:
                    raise RuntimeError("None return in block expression")

                self.exit_scope()

                self.stack = stack
                return return_value

            case NodeEmpty():
                return Unit()

            case NodeDotAccess():
                left = self.eval(expression.lhs)
                return getattr(left, expression.rhs.qualified_name())

            case _:
                raise NotImplementedError(f"{expression=}")

    def enter_scope(self):
        self.stack.append({})

    def exit_scope(self):
        self.stack.pop()

    def push_value_on_stack(self, symbol: Symbol, value: Any):
        self.stack[-1][symbol] = value

    def set_value_on_stack(self, symbol: Symbol, value: Any):
        for frame in reversed(self.stack):
            if symbol in frame:
                frame[symbol] = value
                return

        raise KeyError(f"{symbol} not found in scope")

    def get_value_on_stack(self, symbol: Symbol) -> Any:
        for frame in reversed(self.stack):
            try:
                return frame[symbol]
            except KeyError:
                pass

        raise KeyError(f"{symbol} not found in scope")

    def execute_node(self, line: NodeStmtList):
        return_value = Unit()
        for child in line.children:
            match child:
                case NodeLet():
                    self.push_value_on_stack(
                        child.sym_node.symbol, self.eval(child.expr)
                    )

                case NodeAssignment():
                    assert isinstance(child.lhs, (NodeSymbol, NodeDotAccess)), f"{type(child.lhs)=}"

                    if isinstance(child.lhs, NodeDotAccess):
                        setattr(self.eval(child.lhs.lhs), child.lhs.rhs.qualified_name(), self.eval(child.rhs))
                    else:
                        self.set_value_on_stack(child.lhs.symbol, self.eval(child.rhs))

                case NodeBlockStmt():
                    self.enter_scope()
                    return_value = self.execute_node(child)
                    self.exit_scope()

                case NodeWhileStmt():
                    stack = None
                    while self.eval(child.cond_expr):
                        # copy stack for exections skipping exit_scope
                        stack = self.stack.copy()
                        try:
                            self.execute_node(child.body)
                        except ContinueError:
                            self.stack = stack
                        except BreakError:
                            break

                    if stack is not None:
                        self.stack = stack

                case NodeContinue():
                    raise ContinueError()

                case NodeBreak():
                    raise BreakError()

                case NodeIfStmt():
                    evaluated_cond = self.eval(child.cond_expr)
                    if evaluated_cond:
                        return_value = self.execute_node(child.if_body)
                    else:
                        return_value = self.execute_node(child.else_body)

                case NodeFnDefinition():
                    pass  # nop

                case NodeFnCall():
                    return_value = self.eval(child)

                case NodeReturn():
                    return_value = self.eval(child.expr)
                    logger.debug(f"returning with {return_value}")
                    raise ReturnError(return_value)

                case NodeStructDefinition():
                    self.push_value_on_stack(child.sym, self.construct_serq_struct(child))

                case _:
                    # assume expression
                    return_value = self.eval(child)

            #logger.debug(f"{self.stack=}")

        return return_value

    def execute_module(self, module: Module):
        start = module.ast

        assert start is not None and isinstance(start, NodeStmtList)

        self.enter_scope()
        self.execute_node(start)
