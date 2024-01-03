from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional, Iterator

import hashlib
import textwrap

from lark import Token, Tree

from serqlane.parser import SerqParser


DEBUG = False


RESERVED_KEYWORDS = [
    "and",
    "break",
    "continue",
    "else",
    "fn",
    "if",
    "let",
    "mut",
    "not",
    "or",
    "return",
    "while",
    "struct",
]


class SerqInternalError(Exception): ...


class Node:
    def __init__(self, type: Type) -> None:
        assert type != None and isinstance(type, Type), type
        self.type = type

    def render(self) -> str:
        raise SerqInternalError(f"Render isn't implemented for node type {type(self)}")

class NodeEmpty(Node):
    def __init__(self, type: Type) -> None:
        assert type.kind == TypeKind.unit, "An empty node must have a unit type"
        super().__init__(type)

    def render(self) -> str:
        return ""

class NodeSymbol(Node):
    def __init__(self, symbol: Symbol, type: Type) -> None:
        super().__init__(type)
        self.symbol = symbol

    def render(self) -> str:
        # TODO: Unique global identifier later
        return f"{self.symbol.render()}"

class NodeStmtList(Node):
    def __init__(self, type: Type) -> None:
        super().__init__(type)
        self.children: list[Node] = []

    def add(self, node: Node):
        self.children.append(node)

    def render(self) -> str:
        result = []
        for child in self.children:
            result.append(child.render())
        return "\n".join(result)

class NodeLiteral[T](Node):
    def __init__(self, value: T, type: Type) -> None:
        super().__init__(type) # type either gets converted from literal to actual, gets turned into an error or gets inferred from lhs
        self.value = value

    def render(self) -> str:
        return str(self.value)

class NodeIntLit(NodeLiteral[int]): ...

class NodeFloatLit(NodeLiteral[float]): ...

class NodeBoolLit(NodeLiteral[bool]):
    def render(self) -> str:
        return str(self.value).lower()

class NodeStringLit(NodeLiteral[str]):
    def render(self) -> str:
        return f"\"{self.value}\""

class NodeLet(Node):
    def __init__(self, sym_node: NodeSymbol, expr: Node, type: Type):
        super().__init__(type)
        self.sym_node = sym_node
        self.expr = expr

    def render(self) -> str:
        is_mut = self.sym_node.symbol.mutable
        return f"let {"mut " if is_mut else ""}{self.sym_node.render()}{": " + self.sym_node.type.sym.render()} = {self.expr.render()}"

class NodeAssignment(Node):
    def __init__(self, lhs: Node, rhs: Node, type: Type) -> None:
        super().__init__(type)
        self.lhs = lhs
        self.rhs = rhs

    def render(self) -> str:
        return f"{self.lhs.render()} = {self.rhs.render()}"


class NodeGrouped(Node):
    def __init__(self, inner: Node, type: Type) -> None:
        super().__init__(type)
        self.inner = inner
    
    def render(self) -> str:
        return f"({self.inner.render()})"


class NodeBinaryExpr(Node):
    def __init__(self, lhs: Node, rhs: Node, type: Type) -> None:
        super().__init__(type)
        self.lhs = lhs
        self.rhs = rhs

# TODO: These could be converted into (infix) functions to be stored as op(lhs, rhs) in reimpl
# arith
class NodePlusExpr(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} + {self.rhs.render()})"

class NodeMinusExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} - {self.rhs.render()})"

class NodeMulExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} * {self.rhs.render()})"

class NodeDivExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} / {self.rhs.render()})"

# int only
class NodeModExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} % {self.rhs.render()})"

# logic
class NodeAndExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} and {self.rhs.render()})"

class NodeOrExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} or {self.rhs.render()})"

# comparison
class NodeEqualsExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} == {self.rhs.render()})"

class NodeNotEqualsExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} != {self.rhs.render()})"
    
class NodeLessExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} < {self.rhs.render()})"

class NodeLessEqualsExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} <= {self.rhs.render()})"
    
class NodeGreaterExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} > {self.rhs.render()})"

class NodeGreaterEqualsExpression(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()} >= {self.rhs.render()})"

class NodeDotAccess(Node):
    def __init__(self, lhs: Node, rhs: Symbol) -> None:
        super().__init__(rhs.type)
        self.lhs = lhs
        self.rhs = rhs

    def render(self) -> str:
        return f"{self.lhs.render()}.{self.rhs.render()}"


# others
class NodeBreak(Node):
    def render(self) -> str:
        return "break"

class NodeContinue(Node):
    def render(self) -> str:
        return "continue"


class NodeBlockStmt(NodeStmtList):
    def __init__(self, type: Type) -> None:
        super().__init__(type)

    def render(self) -> str:
        inner = super().render()
        return f"{{\n{textwrap.indent(inner, "  ")}\n}}"

class NodeWhileStmt(Node):
    def __init__(self, cond_expr: Node, body: NodeBlockStmt, type: Type) -> None:
        super().__init__(type)
        self.cond_expr = cond_expr
        self.body = body
    
    def render(self) -> str:
        cond = self.cond_expr.render()
        body = self.body.render()
        return f"while ({cond}) {body}"

class NodeIfStmt(Node):
    def __init__(self, cond_expr: Node, if_body: NodeBlockStmt, else_body: NodeBlockStmt, type: Type) -> None:
        super().__init__(type)
        self.cond_expr = cond_expr
        self.if_body = if_body
        self.else_body = else_body

    def render(self) -> str:
        cond = self.cond_expr.render()
        if_body = self.if_body.render()
        else_body = self.else_body.render()
        return f"if ({cond}) {if_body} else {else_body}"


class NodeReturn(Node):
    def __init__(self, expr: Node, type: Type) -> None:
        super().__init__(type)
        self.expr = expr

    def render(self) -> str:
        return f"return {self.expr.render()}"

class NodeAliasDefinition(Node):
    def __init__(self, sym: Symbol, src: Symbol, type: Type) -> None:
        super().__init__(type)
        self.sym = sym
        self.src = src

    def render(self) -> str:
        return f"alias {self.sym.render()} = {self.src.render()}"

class NodeStructField(Node):
    def __init__(self, sym: Symbol, typ: Type) -> None:
        super().__init__(typ)
        self.sym = sym

    def render(self) -> str:
        return f"{self.sym.render()}: {self.type.sym.render()}"

class NodeStructDefinition(Node):
    def __init__(self, sym: Symbol, fields: list[NodeStructField], type: Type) -> None:
        super().__init__(type)
        self.sym = sym
        self.fields = fields

    def render(self) -> str:
        field_strs = "\n".join(["  " + x.render() for x in self.fields])
        return f"struct {self.sym.render()} {{\n{field_strs}\n}}"

class NodeFnParameters(Node):
    def __init__(self, args: list[tuple[NodeSymbol, Node]]):
        """
        args is a list of (param_sym, type_node)
        type is stored as a node to help with generic instantiation later
        """
        self.args = args

    def render(self) -> str:
        return ", ".join([x[0].render() + ": " + x[1].type.sym.render() for x in self.args])

class NodeFnDefinition(Node):
    def __init__(self, sym: Symbol, params: NodeFnParameters, body: NodeBlockStmt, type: Type) -> None:
        super().__init__(type)
        assert sym.type.kind == TypeKind.function
        self.sym = sym # holds all of the actual type info
        self.sym.definition_node = self # store a reference to self so we can resolve named args
        self.params = params # Need this for named args like foo(a=10)
        self.body = body

    def render(self) -> str:
        return f"fn {self.sym.render()}({self.params.render()}) -> {self.sym.type.sym.render()} {self.body.render()}"

class NodeFnCall(Node):
    def __init__(self, callee: Node, args: list[Node], type: Type) -> None:
        super().__init__(type)
        self.callee = callee
        self.args = args

    def render(self) -> str:
        args = ", ".join([x.render() for x in self.args])
        return f"{self.callee.render()}({args})"


# TODO: Use these, will make later analysis easier
class SymbolKind(Enum):
    variable = auto()
    function = auto()
    parameter = auto()
    type = auto() # TODO: Types
    field = auto()


# TODO: Give every symbol a generation. Deferred function bodies should not be able to look up globals defined after themselves
class Symbol:
    def __init__(self, id: str, name: str, type: Type = None, mutable: bool = False, magic=False) -> None:
        # TODO: Should store the source node, symbol kinds
        self.id = id
        self.name = name
        self.type = type
        self.exported = False
        self.mutable = mutable
        self.definition_node: Node = None
        self.magic = magic

    def qualified_name(self):
        return self.name + self.id

    def __repr__(self) -> str:
        return self.render(debug=True)

    def render(self, debug=DEBUG) -> str:
        # TODO: Use type info to render generics and others
        if debug:
            return f"{self.name}_{self.id}"
        return self.name


class TypeKind(Enum):
    error = auto() # bad type
    infer = auto() # marker type to let nodes know to infer their own types
    magic = auto() # TODO: Get rid of later. Exists to make compiler magics work for debugging
 
    # literal types, haven't yet been matched
    literal_int = auto()
    literal_float = auto()
    literal_bool = auto()
    literal_string = auto()

    unit = auto() # zero sized type

    bool = auto()
    char = auto()

    int8 = auto()
    uint8 = auto()
    int16 = auto()
    uint16 = auto()
    int32 = auto()
    uint32 = auto()
    int64 = auto()
    uint64 = auto()

    float32 = auto()
    float64 = auto()

    pointer = auto() # native sized pointer
    reference = auto() # &T

    string = auto()
    array = auto() # array[T, int]
    static = auto() # TODO: static[T]

    function = auto() # fn(int, float): bool

    alias = auto() # Alias[T]
    distinct = auto() # distinct[T]
    concrete_type = auto() # non-generic concrete Type or fn()
    generic_inst = auto() # fully instantiated generic type Type[int] or fn[int](): generic_inst[concerete_type, generic_type[params]]
    generic_type = auto() # Type[T] or fn[T](): generic_type[params]
    type = auto() # TODO: magic that holds a type itself, not yet in grammar

int_types = frozenset([
    TypeKind.int8,
    TypeKind.uint8,
    TypeKind.int16,
    TypeKind.uint16,
    TypeKind.int32,
    TypeKind.uint32,
    TypeKind.int64,
    TypeKind.uint64,
])

float_types = frozenset([
    TypeKind.float32,
    TypeKind.float64,
])

literal_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_float,
    TypeKind.literal_bool,
    TypeKind.literal_string,
])

# TODO: Other solution based on operator signatures
arith_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_float,
] + list(int_types) + list(float_types))

logical_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_bool,
    TypeKind.bool
] + list(int_types))

free_infer_types = frozenset([
    TypeKind.magic,
    TypeKind.infer,
])

callable_types = frozenset([
    TypeKind.function,
    TypeKind.type,
])


# TODO: Add the other appropriate types
builtin_userspace_types = frozenset(list(int_types) + list(float_types) + [TypeKind.bool, TypeKind.char, TypeKind.string, TypeKind.unit])

class Type:
    def __init__(self, kind: TypeKind, sym: Symbol, data: Any = None) -> None:
        self.kind = kind
        self.data = data # TODO: arbitrary data for now
        self.sym = sym
        # TODO: Add a type id later

    def function_arg_types(self) -> list[Type]:
        assert self.kind == TypeKind.function
        return self.data[0]

    def return_type(self) -> Type:
        assert self.kind == TypeKind.function
        return self.data[1]

    def function_def_args_identical(self, other: Type) -> bool:
        assert self.kind == TypeKind.function and other.kind == TypeKind.function
        my_args = self.function_arg_types()
        other_args = other.function_arg_types()
        if len(my_args) != len(other_args):
            return False # can't be identical if amount is different
        for i in range(0, len(my_args)):
            my_arg = my_args[i]
            other_arg = other_args[i]
            if not my_arg.types_compatible(other_arg):
                return False
        return True

    # TODO: Turn into type_relation. literal<->concrete means uninstantiated_literal
    def types_compatible(self, other: Type) -> bool:
        """
        other is always the target
        """
        if self.kind in free_infer_types or other.kind in free_infer_types:
            return True

        # TODO: Match variant, like generic inst of generic type
        match self.kind:
            case TypeKind.error:
                return False
            
            # TODO: Not sure what to do about these
            case TypeKind.unit:
                return self.kind == other.kind
            
            case TypeKind.function:
                raise NotImplementedError() # TODO

            # magic types

            # TODO: Can you match sets?
            case TypeKind.char | TypeKind.pointer:
                return self.kind == other.kind
            case TypeKind.bool:
                return self.kind == other.kind or other.kind == TypeKind.literal_bool    
            case TypeKind.int8 | TypeKind.uint8 | TypeKind.int16 | TypeKind.uint16 | TypeKind.int32 | TypeKind.uint32 | TypeKind.int64 | TypeKind.uint64:
                return self.kind == other.kind or other.kind == TypeKind.literal_int
            case TypeKind.float32 | TypeKind.float64:
                return self.kind == other.kind or other.kind == TypeKind.literal_float
            case TypeKind.string:
                return self.kind == other.kind or other.kind == TypeKind.literal_string

            case TypeKind.reference:
                if self.kind != other.kind:
                    return False
                return self.data.compare(other.data)
            
            case TypeKind.array:
                if self.kind != other.kind:
                    return False
                return self.data[0].compare(other.data[0]) and self.data[1].compare(other.data[0])
            
            case TypeKind.static:
                # TODO: static is allowed to be turned into the corresponding non-static version but not vice versa
                raise SerqInternalError("statics aren't ready yet")

            case TypeKind.alias:
                # TODO: Generic aliases. Need to map potentially new generic params around and then reduce the type if required
                raise SerqInternalError("aliases aren't ready yet")
            
            case TypeKind.distinct:
                # TODO: distinct generics are tough
                raise SerqInternalError("distincts aren't ready yet")
            

            # user types

            case TypeKind.generic_inst:
                raise SerqInternalError("generic instances aren't ready yet")

            case TypeKind.generic_type:
                raise SerqInternalError("generic types aren't ready yet")

            case TypeKind.type | TypeKind.concrete_type:
                if id(self) == id(other): # TODO: It should NOT use python id, should use a concrete id or something. Need a type cache for that
                    return True
                # TODO: If this fails even though the types should be the same the cache got messed up somehow


            # literals

            case TypeKind.literal_bool:
                return other.kind in {TypeKind.literal_bool, TypeKind.bool}
            case TypeKind.literal_int:
                return other.kind in int_types or other.kind == TypeKind.literal_int
            case TypeKind.literal_float:
                return other.kind in float_types or other.kind == TypeKind.literal_float
            case TypeKind.literal_string:
                return other.kind in {TypeKind.literal_string, TypeKind.string}

            case _:
                raise SerqInternalError(f"Unimplemented type comparison: {self.kind}")

    def instantiate_literal(self, graph: ModuleGraph) -> Type:
        """
        Turns a literal into a concrete type
        """
        assert self.kind in literal_types
        match self.kind:
            case TypeKind.literal_int:
                return graph.builtin_scope.lookup_type("int64")
            case TypeKind.literal_float:
                return graph.builtin_scope.lookup_type("float64")
            case TypeKind.literal_bool:
                return graph.builtin_scope.lookup_type("bool")
            case TypeKind.literal_string:
                return graph.builtin_scope.lookup_type("string")
            case _:
                raise SerqInternalError(f"Forgot a literal type: {self.kind}")

    def render(self) -> str:
        # TODO: match on sets?
        if self.kind in builtin_userspace_types or self.kind in literal_types:
            return self.kind.name
        elif self.kind == TypeKind.function:
            args = ", ".join([x.render() for x in self.data[0]])
            return f"fn({args}): {self.data[1].render()}"
        elif self.kind == TypeKind.type:
            return self.sym.definition_node.render()
        else:
            raise SerqInternalError(f"Render isn't implemented for type kind {self.kind}")


class Scope:
    def __init__(self, graph: ModuleGraph) -> None:
        self._local_syms: list[Symbol] = []
        self.parent: Scope = None
        self.module_graph = graph # TODO: Get rid of builtin hack

    def iter_syms(self, name: Optional[str] = None) -> Iterator[Symbol]:
        # prefer magics
        if self.module_graph.builtin_scope != self:
            for sym in self.module_graph.builtin_scope.iter_syms(name):
                if name == None or sym.name == name:
                    yield sym

        for sym in self._local_syms:
            if name == None or sym.name == name:
                yield sym
        if self.parent != None:
            yield from self.parent.iter_syms(name)

    def iter_function_defs(self, name: Optional[str] = None) -> Iterator[Symbol]:
        for sym in self.iter_syms(name):
            if sym.type.kind in callable_types:
                yield sym

    def _lookup_impl(self, name: str, shallow=False) -> Symbol:
        for symbol in self._local_syms:
            if symbol.name == name:
                return symbol
        if shallow or self.parent == None:
            # TODO: Report error in module
            return None
        return self.parent._lookup_impl(name)

    def lookup(self, name: str, shallow=False) -> Optional[Symbol]:
        # Must be unambiguous, can return an unexported symbol. Checked at calltime
        magic = self.module_graph.builtin_scope._lookup_impl(name, shallow=True) # TODO: hack
        if magic:
            return magic
        return self._lookup_impl(name, shallow)

    def lookup_type(self, name: str, shallow=False) -> Optional[Type]:
        # helper for trivial case of sym.type
        sym = self.lookup(name, shallow=shallow)
        if sym != None:
            return sym.type

    def inject(self, sym: Symbol):
        self._local_syms.append(sym)

    def put(self, name: str, checked=True, shallow=False) -> Symbol:
        assert type(name) == str
        if checked and self.lookup(name, shallow=shallow): raise ValueError(f"redefinition of {name}")

        if name in RESERVED_KEYWORDS:
            raise ValueError(f"Cannot use reserved keyword `{name}` as a symbol name")

        result = Symbol(self.module_graph.sym_id_gen.next(), name=name)
        self.inject(result)
        return result
    
    def put_function(self, name: str, type: Type) -> Symbol:
        assert type.kind == TypeKind.function
        for fn in self.iter_function_defs(name):
            if fn.type.function_def_args_identical(type):
                raise ValueError(f"Redefinition of function {name}")
        sym = self.put(name, checked=False)
        sym.type = type
        return sym

    def put_magic(self, name: str) -> Symbol:
        assert type(name) == str
        if self.lookup(name, shallow=True): raise SerqInternalError(f"redefinition of magic sym: {name}")
        result = Symbol(self.module_graph.sym_id_gen.next(), name=name, magic=True)
        self._local_syms.append(result)
        return result

    def put_magic_function(self, name: str, typ: Type) -> Symbol:
        assert type(name) == str
        sym = self.put_function(name, typ)
        sym.magic = True
        return sym

    def put_builtin_type(self, kind: TypeKind) -> Symbol:
        # TODO: Get rid of this hack
        sym = self.put_magic(kind.name)
        sym.type = Type(kind=kind, sym=sym)
        return sym
    
    def put_type(self, name: str) -> Symbol:
        sym = self.put(name)
        sym.type = Type(kind=TypeKind.type, sym=sym)
        return sym

    def put_let(self, name: str, mutable=False, checked=True, shallow=False) -> Symbol:
        sym = self.put(name, checked=checked, shallow=shallow)
        sym.mutable = mutable
        return sym

    def make_child(self) -> Scope:
        res = Scope(self.module_graph)
        res.parent = self
        return res

    def merge(self, other: Scope):
        ...


class CompCtx:
    def __init__(self, module: Module, graph: ModuleGraph) -> None:
        self.module = module
        self.graph = graph
        self.current_scope = self.module.global_scope
        self.in_loop_counter = 0
        
        # deferred body mode
        self.handling_deferred_fn_body = False
        self.current_deferred_ret_type: Optional[Type] = None

    def open_scope(self):
        self.current_scope = self.current_scope.make_child()

    def close_scope(self):
        self.current_scope = self.current_scope.parent

    def defer_fn_body(self, sym: Symbol, body: Tree):
        self.module.deferred_fn_bodies.append((sym, body))


    def get_infer_type(self) -> Type:
        return Type(kind=TypeKind.infer, sym=None)

    def get_unit_type(self) -> Type:
        return self.current_scope.lookup_type("unit", shallow=True)


    def statement(self, tree: Tree, expected_type: Type) -> Node:
        assert tree.data == "statement", tree.data
        assert len(tree.children) == 1, f"{len(tree.children)} --- {tree.children=}"

        child = tree.children[0]
        match child.data:
            case "fn_definition":
                return self.fn_definition(child, expected_type)
            case "expression":
                return self.expression(child, expected_type)
            case "struct_definition":
                return self.struct_definition(child, expected_type)
            case "alias_definition":
                return self.alias_definition(child, expected_type)
            case "let_stmt":
                return self.let_stmt(child, expected_type)
            case "return_stmt":
                return self.return_stmt(child)
            case "assignment":
                return self.assignment(child, expected_type)
            case "if_stmt":
                return self.if_stmt(child, expected_type)
            case "while_stmt":
                return self.while_stmt(child, expected_type)
            case "break_stmt" | "continue_stmt":
                return self.handle_break_or_continue(child, expected_type)
            case "block_stmt": 
                return self.handle_block(child, expected_type)
            case _:
                raise SerqInternalError(f"Unimplemented statement type: {child.data}")


    def handle_break_or_continue(self, tree: Tree, expected_type: Type) -> NodeBreak | NodeContinue:
        assert tree.data in ["break_stmt", "continue_stmt"], tree.data
        assert expected_type.kind == TypeKind.unit
        assert len(tree.children) == 0
        if self.in_loop_counter < 1:
            raise ValueError("Break or continue outside of a loop")
        if tree.data == "break_stmt":
            return NodeBreak(self.get_unit_type())
        elif tree.data == "continue_stmt":
            return NodeContinue(self.get_unit_type())


    def while_stmt(self, tree: Tree, expected_type: Type) -> NodeWhileStmt:
        assert tree.data == "while_stmt", tree.data
        assert expected_type.kind == TypeKind.unit
        # This has to use the outer scope, so a new scope is only opened once this has been checked in full
        while_cond = self.expression(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert while_cond.type.kind == TypeKind.bool

        # block_stmt opens a scope
        self.in_loop_counter += 1 # needed to check if break and continue are valid
        body = self.handle_block(tree.children[1], self.get_unit_type())
        self.in_loop_counter -= 1

        return NodeWhileStmt(while_cond, body, self.get_unit_type())

    def if_stmt(self, tree: Tree, expected_type: Type) -> NodeIfStmt:
        assert tree.data == "if_stmt", tree.data
        assert expected_type.kind == TypeKind.unit # TODO: if expressions are not unit, need to guarantee valid else branch
        # Same scoping story as in while_stmt
        if_cond = self.expression(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert if_cond.type.kind == TypeKind.bool
        if_body = self.handle_block(tree.children[1], self.get_unit_type())

        else_body = None
        if tree.children[2] != None:
            else_body = self.handle_block(tree.children[2], self.get_unit_type())
        else:
            # Always inject an empty else case if none is provided
            else_body = NodeBlockStmt(self.get_unit_type())
        return NodeIfStmt(if_cond, if_body, else_body, self.get_unit_type()) # TODO: Pass along branch types once they exist


    def handle_block(self, tree: Tree, expected_type: Type) -> NodeBlockStmt:
        assert tree.data in ["block_stmt", "block_expression"], tree.data
        self.open_scope()
        if len(tree.children) == 0:
            return NodeBlockStmt(self.get_unit_type())

        # Assume unit type if nothing is expected, fixed later
        # Have to be very careful with symbols, we do not want to use one that only exists later
        result = NodeBlockStmt(expected_type if expected_type != None else self.get_unit_type())

        (tree_children, last_child) = (tree.children[0:len(tree.children)-1], tree.children[-1])
        for child in tree_children:
            # All but the last have to be unit typed
            result.add(self.statement(child, self.get_unit_type()))

        if expected_type != None:
            result.add(self.statement(last_child, expected_type))
            # TODO: Get rid of check here once shadow syms are in
            assert expected_type.types_compatible(result.children[-1].type), f"Expected type {expected_type.sym.render()} for block but got {result.children[-1].type.sym.render()}"
            result.type = result.children[-1].type
        elif len(result.children) > 0:
            result.add(self.statement(last_child, None))
            result.type = result.children[-1].type

        self.close_scope()
        return result

    def grouped_expression(self, tree: Tree, expected_type: Type) -> NodeGrouped:
        assert tree.data == "grouped_expression", tree.data
        inner = self.expression(tree.children[0], expected_type)
        return NodeGrouped(inner, inner.type)


    def handle_literal(self, tree: Tree, expected_type: Type, lookup_name: str, literal_kind: TypeKind, node_type: type[NodeLiteral], conv_fn):
        if len(tree.children) > 0:
            value = conv_fn(tree.children[0].value)
        
        # empty literals i.e. "" or []
        else:
            # match so it's easy to add new ones
            match literal_kind:
                case TypeKind.literal_string:
                    value = ""
                case _:
                    raise SerqInternalError("Unsupported empty literal")

        if literal_kind is TypeKind.literal_string:
            value = self.resolve_escape_sequence(value)

        if expected_type is not None:
            if expected_type.kind in free_infer_types:
                expected_type = self.current_scope.lookup_type(lookup_name, shallow=True)
            else:
                assert expected_type.types_compatible(Type(literal_kind, sym=None))
            return node_type(value=value, type=expected_type)
        else:
            return node_type(value=value, type=Type(literal_kind, sym=None))

    @staticmethod
    def resolve_escape_sequence(value: str) -> str:
        return value.replace('\\"', '"')

    def integer(self, tree: Tree, expected_type: Type) -> NodeIntLit:
        assert tree.data == "integer", tree.data
        return self.handle_literal(tree, expected_type, "int64", TypeKind.literal_int, NodeIntLit, int)
    
    def decimal(self, tree: Tree, expected_type: Type) -> NodeFloatLit:
        assert tree.data == "decimal", tree.data
        return self.handle_literal(tree, expected_type, "float64", TypeKind.literal_float, NodeFloatLit, float)
    
    def bool(self, tree: Tree, expected_type: Type) -> NodeBoolLit:
        assert tree.data == "bool", tree.data
        return self.handle_literal(tree, expected_type, "bool", TypeKind.literal_bool, NodeBoolLit, lambda x: x == "true")
    
    def string(self, tree: Tree, expected_type: Type) -> NodeStringLit:
        assert tree.data == "string", tree.data
        return self.handle_literal(tree, expected_type, "string", TypeKind.literal_string, NodeStringLit, str)


    def binary_expression(self, tree: Tree, expected_type: Type) -> NodeBinaryExpr:
        assert tree.data == "binary_expression", tree.data
        lhs = self.expression(tree.children[0], None)
        # TODO: Can the symbol be captured somehow?
        op = tree.children[1].data.value
        # a dot expression does not work by the same type rules
        if op != "dot":
            rhs = self.expression(tree.children[2], None)

            # TODO: Dot expr
            if not lhs.type.types_compatible(rhs.type):
                # TODO: Error reporting
                tl = lhs.type.instantiate_literal(self.graph) if lhs.type.kind in literal_types else lhs.type
                tr = rhs.type.instantiate_literal(self.graph) if rhs.type.kind in literal_types else rhs.type
                raise ValueError(f"Incompatible values in binary expression: `{lhs.render()}`:{lhs.render()} {op} `{tl.sym.render()}`:{tr.sym.render()}")
            
            # Type coercion. Revisits "broken" nodes and tries to apply the new info on them 
            # TODO: Parts of this should probably be pulled into a new function
            expr_type: Type = lhs.type if lhs.type.kind not in literal_types else rhs.type

            # if there is a known type, spread it around
            if lhs.type.kind in literal_types and rhs.type.kind not in literal_types:
                lhs = self.expression(tree.children[0], rhs.type)
            elif rhs.type.kind in literal_types and lhs.type.kind not in literal_types:
                rhs = self.expression(tree.children[2], lhs.type)
            
            # both literal
            else:
                # if possible we ask for help from expected_type
                if expected_type != None and expr_type.types_compatible(expected_type):
                    lhs = self.expression(tree.children[0], expected_type)
                    rhs = self.expression(tree.children[2], expected_type)
                    assert lhs.type.types_compatible(rhs.type)
                    expr_type = lhs.type

                # no expectation, let them infer their own types
                elif expected_type == None:
                    lhs = self.expression(tree.children[0], self.get_infer_type())
                    rhs = self.expression(tree.children[2], self.get_infer_type())
                    assert lhs.type.types_compatible(rhs.type)
                    expr_type = lhs.type
                
                # there is an expected type but the expression isn't compatible with it
                else:
                    # Assume it's fine, it should be checked outside of here
                    pass
                    #assert False, f"{expected_type.kind=}    {expr_type.kind}"

        match op:
            case "plus":
                assert expr_type.kind in arith_types
                return NodePlusExpr(lhs, rhs, type=expr_type)
            case "minus":
                assert expr_type.kind in arith_types
                return NodeMinusExpression(lhs, rhs, type=expr_type)
            case "star":
                assert expr_type.kind in arith_types
                return NodeMulExpression(lhs, rhs, type=expr_type)
            case "slash":
                assert expr_type.kind in arith_types
                return NodeDivExpression(lhs, rhs, type=expr_type)
            
            case "modulus":
                assert expr_type.kind in int_types
                return NodeModExpression(lhs, rhs, type=expr_type)
            
            case "and":
                assert expr_type.kind in logical_types
                return NodeAndExpression(lhs, rhs, type=expr_type)
            case "or":
                assert expr_type.kind in logical_types
                return NodeOrExpression(lhs, rhs, type=expr_type)
            
            # TODO: Is this version of ensure_types correct here?
            case "equals":
                return NodeEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))
            case "not_equals":
                return NodeNotEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))
            case "less":
                # TODO: Ordinal types.
                assert expr_type.kind in arith_types
                return NodeLessExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))
            case "lesseq":
                assert expr_type.kind in arith_types
                return NodeLessEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))
            case "greater":
                assert expr_type.kind in arith_types
                return NodeGreaterExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))
            case "greatereq":
                assert expr_type.kind in arith_types
                return NodeGreaterEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool"))

            case "dot":
                # TODO: Has to use the type of rhs for node type, can only be done once types and field lookup exist
                lhs = self.expression(tree.children[0], None)
                assert lhs.type.kind == TypeKind.type, f"Dot operations for type {lhs.type.sym.render()} are not yet allowed" # TODO: Allow any type here, required for universal function calling syntax
                rhs = tree.children[2].children[0].value

                assert isinstance(lhs.type.sym.definition_node, NodeStructDefinition), "Dot operations are currently only ready for structs"
                matching_field_sym = None
                # TODO: Do a scope-esque lookup instead. Things like inheritance may silently add more things later
                for field in lhs.type.sym.definition_node.fields:
                    if field.sym.name == rhs:
                        matching_field_sym = field.sym
                        break
                assert matching_field_sym != None, f"Could not find {rhs} for {lhs.type.sym.render()}"

                return NodeDotAccess(lhs, matching_field_sym)
            case _:
                raise SerqInternalError(f"Unimplemented binary op: {op}")


    def identifier(self, tree: Tree, expected_type: Type) -> NodeSymbol:
        assert tree.data == "identifier", tree.data
        val = tree.children[0].value
        sym = self.current_scope.lookup(val)
        if sym:
            if expected_type != None:
                assert sym.type.types_compatible(expected_type)
            return NodeSymbol(sym, type=sym.type)
        # TODO: Error reporting
        raise ValueError(f"Bad identifier: {val}")

    def user_type(self, tree: Tree, expected_type: Type) -> NodeSymbol:
        assert tree.data in ["user_type", "return_user_type"], tree.data
        return self.identifier(tree.children[0], None) # TODO: Enforce `type` metatype

    def assignment(self, tree: Tree, expected_type: Type) -> NodeAssignment:
        assert tree.data == "assignment", tree.data
        # TODO: Assignments could technically act as expressions too
        assert expected_type.kind == TypeKind.unit

        lhs = self.expression(tree.children[0], None)
        rhs = self.expression(tree.children[1], lhs.type)
        assert lhs.type.types_compatible(rhs.type)

        def report_immutable(sym: Symbol):
            raise ValueError(f"{sym.name} is not mutable")

        match lhs:
            case NodeSymbol():
                if not lhs.symbol.mutable:
                    report_immutable(lhs.symbol)
            case NodeDotAccess():
                leftmost = lhs.lhs
                while isinstance(leftmost, NodeDotAccess):
                    leftmost = leftmost.lhs
                assert isinstance(leftmost, NodeSymbol)
                if not leftmost.symbol.mutable:
                    report_immutable(leftmost.symbol)
            case _:
                raise NotImplementedError()
        
        return NodeAssignment(lhs, rhs, self.get_unit_type())

    def let_stmt(self, tree: Tree, expected_type: Type) -> NodeLet:
        assert tree.data == "let_stmt", tree.data
        mut_node = tree.children[0]

        # currently the only modifier
        is_mut = isinstance(mut_node, Token)
        if isinstance(mut_node, Token):
            assert mut_node.type == "MUT"

        f = int(is_mut)
        ident_node = tree.children[f]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value

        f += 1

        type_sym = None
        if tree.children[f].data == "user_type":
            # we have a user provided type node
            type_tree = tree.children[f]
            # TODO: Return an empty node with empty type
            # TODO: Have expected type be `type` metatype
            type_sym = self.user_type(type_tree, None)
            f += 1
        
        val_node_expected_type = type_sym.type if type_sym != None else self.get_infer_type()
        val_node = self.expression(tree.children[f], val_node_expected_type)
        if val_node.type.kind == TypeKind.unit:
            raise ValueError(f"Type `{val_node.type.kind.name}` is not valid for `let`")

        resolved_type = None
        if type_sym != None:
            resolved_type = type_sym.type
            # check types for compatibility
            if not val_node.type.types_compatible(resolved_type):
                # TODO: Error reporting
                raise ValueError(f"Variable type {type_sym.symbol.name} is not compatible with value of type {val_node.type.sym.render()}")
            else:
                assert val_node.type.kind not in literal_types
            # no need to resem the node, it should already be with the current type resolution
        else:
            # infer type from value
            # TODO: Instantiate types, for now only literals            
            if val_node.type.kind in builtin_userspace_types or val_node.type.kind == TypeKind.type:
                resolved_type = val_node.type
            else:
                # Literals infer their own types to the default if told to do so
                assert False

        f += 1
        assert len(tree.children) == f

        sym = self.current_scope.put_let(ident, mutable=is_mut)
        sym.type = resolved_type
        sym = NodeSymbol(sym, sym.type)
        return NodeLet(
            sym_node=sym,
            expr=val_node,
            type=self.get_unit_type()
        )

    def return_stmt(self, tree: Tree) -> NodeReturn:
        assert tree.data == "return_stmt", tree.data
        assert self.handling_deferred_fn_body, "Return outside of a function"
        # TODO: Make sure this passes type checks
        expr = None
        if tree.children[0] != None:
            expr = self.expression(tree.children[0], self.current_deferred_ret_type)
        else:
            expr = NodeEmpty(self.current_deferred_ret_type)
        assert self.current_deferred_ret_type.types_compatible(expr.type), f"Incompatible return({expr.type.sym.render()}) for function type({self.current_deferred_ret_type.render()})"
        return NodeReturn(expr=expr, type=self.get_unit_type())

    def alias_definition(self, tree: Tree, expected_type: Type) -> NodeAliasDefinition:
        assert tree.data == "alias_definition", tree.data
        assert expected_type.kind == TypeKind.unit

        src = self.identifier(tree.children[1], None)

        alias_name = tree.children[0].children[0].value
        alias_sym = self.current_scope.put(alias_name)
        alias_sym.type = src.type
        alias_sym.mutable = src.symbol.mutable
        alias_sym.magic = src.symbol.magic
        # TODO: Exporting aliases
        # alias_sym.exported = ...

        res_node = NodeAliasDefinition(alias_sym, src.symbol, self.get_unit_type())
        alias_sym.definition_node = res_node
        return res_node

    def struct_field(self, tree: Tree, expected_type: Type) -> NodeStructField:
        assert tree.data == "struct_field", tree.data
        assert expected_type.kind == TypeKind.unit

        ident = tree.children[0].children[0].value
        sym = self.current_scope.put_let(ident, checked=True, shallow=True)

        # TODO: Assert that this is actually a type sym. Current system isn't ready yet
        typ_sym_node = self.user_type(tree.children[1], None)

        sym.type = typ_sym_node.type

        return NodeStructField(sym, typ_sym_node.symbol.type)

    def struct_definition(self, tree: Tree, expected_type: Type) -> NodeStructDefinition:
        assert tree.data == "struct_definition", tree.data
        assert expected_type.kind == TypeKind.unit
        ident = tree.children[0].children[0].value
        sym = self.current_scope.put_type(ident)

        fields = []
        if tree.children[1] != None:
            self.open_scope()
            for field_node in tree.children[1:]:
                # TODO: Recursion check. Recursion is currently unrestricted and permits infinite recursion
                #   For proper recursion handling we also gotta ensure we don't try to access a type that is currently being processed like will be the case with mutual recursion
                field = self.struct_field(field_node, self.get_unit_type())
                fields.append(field)
            self.close_scope()

        def_node = NodeStructDefinition(sym, fields, self.get_unit_type())
        sym.definition_node = def_node
        return def_node
    
    def handle_deferred_fn_body(self, tree: Tree, sym: Symbol) -> NodeBlockStmt:
        self.current_deferred_ret_type = sym.type.return_type()

        # isolate params again
        self.open_scope()
        d: NodeFnDefinition = None
        for param in sym.definition_node.params.args:
            self.current_scope.inject(param[0].symbol)

        body = self.handle_block(tree, self.get_infer_type())

        # TODO: Revisit this segment once symbol shadowing is in.
        # All of this should be handled by a transformation sym
        if len(body.children) > 0:
            last_body_node = body.children[-1]
            if last_body_node.type.kind == TypeKind.unit and not sym.type.return_type().kind == TypeKind.unit and not isinstance(last_body_node, NodeReturn):
                assert False, f"Returning a unit type for expected type {sym.type.return_type().sym.render()} is not permitted"
            elif isinstance(last_body_node, NodeReturn):
                pass # return is already checked
            elif last_body_node.type.kind in literal_types:
                body = self.handle_block(tree.children[3], body)
                last_body_node = body.children[-1]
                if last_body_node.type.kind in builtin_userspace_types:
                    assert last_body_node.type.types_compatible(sym.type.return_type()), f"Invalid return expression type {last_body_node.type.sym.render()} for return type {sym.type.return_type().sym.render()}"
                body.children[-1] = NodeReturn(last_body_node, self.get_unit_type())
            else:
                assert last_body_node.type.types_compatible(sym.type.return_type()), f"Invalid return expression type {last_body_node.type.sym.render()} for return type {sym.type.return_type().sym.render()}"
                if last_body_node.type.kind == TypeKind.unit:
                    body.children.append(NodeReturn(NodeEmpty(self.get_unit_type()), self.get_unit_type()))
                else:
                    body.children[-1] = NodeReturn(last_body_node, self.get_unit_type())
        else:
            assert sym.type.return_type().kind == TypeKind.unit
            body.children.append(NodeReturn(NodeEmpty(self.get_unit_type()), self.get_unit_type()))

        self.close_scope()

        return body

    def fn_definition_args(self, tree: Tree, expected_type: Type) -> NodeFnParameters:
        assert tree.data == "fn_definition_args", tree.data
        assert expected_type.kind == TypeKind.unit

        params = []

        if len(tree.children) == 1 and tree.children[0] is None:
            return NodeFnParameters(params)

        for child in tree.children:
            assert child.data == "fn_definition_arg"
            # TODO: Mutable args
            ident = child.children[0].children[0].value
            sym = self.current_scope.put_let(ident, shallow=True) # effectively a let that permits shallow shadowing
            type_node = self.user_type(child.children[1], None)
            sym.type = type_node.symbol.type # TODO: This is not clean at all
            params.append((NodeSymbol(sym, sym.type), type_node))

        return NodeFnParameters(params)

    def fn_definition(self, tree: Tree, expected_type: Type) -> NodeFnDefinition:
        assert tree.data == "fn_definition", tree.data
        assert expected_type.kind == TypeKind.unit

        ident_node = tree.children[0]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value

        # must open a scope here to isolate the params
        self.open_scope()
        
        args_node = self.fn_definition_args(tree.children[1], self.get_unit_type())
        ret_type_node = NodeSymbol(self.get_unit_type().sym, self.get_unit_type())
        ret_type = ret_type_node.type
        if tree.children[2] != None:
            ret_type_node = self.user_type(tree.children[2], None)
            ret_type = ret_type_node.type # TODO: Fix this nonsense, type should be `type`, not whatever the type evaluates to

        # TODO: Use a type cache to load function with the same type from it for easier matching
        fn_type = Type(
            kind=TypeKind.function,
            sym=None,
            data=([x[1].type for x in args_node.args], ret_type)
        )

        self.close_scope()

        # The sym must be created here to make recursive calls work without polluting the arg scope
        sym = self.current_scope.put_function(ident, fn_type)
        fn_type.sym = sym

        # TODO: Make this work for generics later
        self.defer_fn_body(sym, tree.children[3])

        sym.type = fn_type

        res = NodeFnDefinition(sym, args_node, None, self.get_unit_type())
        sym.definition_node = res
        return res

    def fn_call_expr(self, tree: Tree, expected_type: Type) -> NodeFnCall:
        assert tree.data == "fn_call_expr", tree.data
        # TODO: Once overloads/methods are in, this must scan visible symbols and retrieve a list. Handled in identifier though
        passed_arg_count = len(tree.children[1].children) if tree.children[1].children[0] != None else 0

        callee = None
        params = []
        if len(tree.children[0].children) > 0 and tree.children[0].children[0].data == "identifier":
            # TODO: Better handling, use a candidate node instead
            for fn in self.current_scope.iter_function_defs(tree.children[0].children[0].children[0].value):
                if fn.type.kind == TypeKind.type:
                    assert passed_arg_count == 0, "Calling a struct with arguments is not yet supported" # TODO 
                    callee = NodeSymbol(fn, fn.type)
                    break
                else:
                    if passed_arg_count != len(fn.type.function_arg_types()):
                        continue
                    matches = True
                    # TODO: This will be wrong once optional args are in
                    for i in range(0, passed_arg_count):
                        # TODO: DO NOT USE TRY EXCEPT. Only here because shadows aren't in yet
                        try:
                            resolved_arg = self.expression(tree.children[1].children[i], fn.type.function_arg_types()[i])
                        except:
                            matches = False
                            break

                        if not resolved_arg.type.types_compatible(fn.type.function_arg_types()[i]):
                            matches = False
                            break
                        params.append(resolved_arg)
                    if matches:
                        callee = NodeSymbol(fn, fn.type)
                        break
                params = []
        else:
            callee = self.expression(tree.children[0], None)
        # TODO: allow non-symbol calls i.e. function pointers
        assert callee != None, f"No matching overload found for {tree.children[0].children[0].children[0]}"
        assert isinstance(callee, NodeSymbol), "can only call symbols"

        ret_type = None
        if callee.type.kind == TypeKind.type:
            ret_type = callee.type
        else:
            ret_type = callee.symbol.type.return_type()
            if ret_type == None:
                ret_type = self.current_scope.lookup_type("unit", shallow=True)

        return NodeFnCall(
            callee=callee,
            args=params,
            type=ret_type
        )

    def expression(self, tree: Tree, expected_type: Type) -> Node:
        assert tree.data == "expression", tree.data
        # TODO: Handle longer expressions?
        assert len(tree.children) == 1
        result = []
        tree: Tree = tree
        for child in tree.children:
            match child.data:
                case "fn_call_expr":
                    result.append(self.fn_call_expr(child, expected_type))
                case "binary_expression":
                    result.append(self.binary_expression(child, expected_type))
                case "block_expression":
                    result.append(self.handle_block(child, expected_type))
                case "grouped_expression":
                    result.append(self.grouped_expression(child, expected_type))

                # TODO: Literals should expect their literal types, fixed later by sym transformations once shadows exist
                case "bool":
                    result.append(self.bool(child, expected_type))
                case "integer":
                    result.append(self.integer(child, expected_type))
                case "decimal":
                    result.append(self.decimal(child, expected_type))
                case "string":
                    result.append(self.string(child, expected_type))
                
                case "identifier":
                    result.append(self.identifier(child, expected_type))
                case _:
                    raise SerqInternalError(f"Unimplemented expression type: {child.data}")
        if len(result) == 1:
            return result[0]
        else:
            assert False

    def start(self, tree: Tree, expected_type: Type) -> NodeStmtList:
        assert tree.data == "start", tree.data
        result = NodeStmtList(self.current_scope.lookup_type("unit", shallow=True))
        for child in tree.children:
            node = self.statement(child, self.get_unit_type())
            result.add(node)
        return result


class Module:
    def __init__(self, name: str, id: int, contents: str, graph: ModuleGraph) -> None:
        self.graph = graph
        # TODO: Naming conflicts
        self.imported_modules: list[Module] = []

        self.name = name
        self.global_scope = Scope(graph)
        self.id = id
        self.hash = hashlib.md5(contents.encode()).digest()
        self.lark_tree = SerqParser().parse(contents, display=False)
        self.ast: Node = None
        # mapping of (name, dict[params]) -> Type
        self.generic_cache: dict[tuple[str, dict[Symbol, Type]], Type] = {}

        # TODO: Generics should use the same deferred trick, but they should not be removed from the deferred list
        self.deferred_fn_bodies: list[tuple[Symbol, Tree]] = []

    def lookup_toplevel(self, name: str) -> Optional[Symbol]:
        return self.global_scope.lookup(name, shallow=True)


class IdGen:
    def __init__(self) -> None:
        self._idx = 0 # TODO: choose a smarter way later

    def next(self) -> str:
        res = str(self._idx)
        self._idx += 1
        return res

class ModuleGraph:
    def __init__(self) -> None:
        self.modules: dict[str, Module] = {} # TODO: Detect duplicates based on hash, reduce workload
        self._next_id = 0 # TODO: Generate in a smarter way
        self.sym_id_gen = IdGen()

        # TODO: Use a type cache instead of scope hack
        self.builtin_scope = Scope(self)

        unit_type_sym = self.builtin_scope.put_builtin_type(TypeKind.unit)

        self.builtin_scope.put_builtin_type(TypeKind.bool)
        self.builtin_scope.put_builtin_type(TypeKind.char)

        self.builtin_scope.put_builtin_type(TypeKind.int8)
        self.builtin_scope.put_builtin_type(TypeKind.uint8)
        self.builtin_scope.put_builtin_type(TypeKind.int16)
        self.builtin_scope.put_builtin_type(TypeKind.uint16)
        self.builtin_scope.put_builtin_type(TypeKind.int32)
        self.builtin_scope.put_builtin_type(TypeKind.uint32)
        self.builtin_scope.put_builtin_type(TypeKind.int64)
        self.builtin_scope.put_builtin_type(TypeKind.uint64)

        self.builtin_scope.put_builtin_type(TypeKind.float32)
        self.builtin_scope.put_builtin_type(TypeKind.float64)

        self.builtin_scope.put_builtin_type(TypeKind.pointer)

        # TODO
        self.builtin_scope.put_builtin_type(TypeKind.string)
        self.builtin_scope.put_builtin_type(TypeKind.array)
        self.builtin_scope.put_builtin_type(TypeKind.static)


        # TODO: hack
        magic_sym = Symbol("-1", "magic")
        magic_type = Type(TypeKind.magic, magic_sym)
        magic_sym.type = magic_type

        dbg_sym_type = Type(TypeKind.function, None, ([magic_type], unit_type_sym.type))
        dbg_sym = self.builtin_scope.put_magic_function("dbg", dbg_sym_type)
        dbg_sym_type.sym = dbg_sym


    def load(self, name: str, file_contents: str) -> Module:
        assert name not in self.modules
        mod = Module(name, self._next_id, file_contents, self)
        self._next_id += 1
        self.modules[name] = mod
        # TODO: Make sure the module isn't already being processed
        ctx = CompCtx(mod, self)
        ast: NodeStmtList = ctx.start(mod.lark_tree, None)

        # TODO: Check type cohesion
        ctx.handling_deferred_fn_body = True
        for fn in mod.deferred_fn_bodies:
            body = ctx.handle_deferred_fn_body(fn[1], fn[0])
            fn[0].definition_node.body = body
        ctx.handling_deferred_fn_body = False

        mod.ast = ast
        return mod
