from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional, Iterator

import hashlib
import textwrap

import lark.visitors
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
]

class Node:
    def __init__(self, type: Type) -> None:
        assert type != None and isinstance(type, Type)
        self.type = type

    def render(self) -> str:
        raise NotImplementedError(f"{type(self)}")

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
        return f"fn {self.sym.render()}({self.params.render()}) -> {self.sym.type.data[1].symbol.render()} {self.body.render()}"

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

    s8 = auto()
    u8 = auto()
    s16 = auto()
    u16 = auto()
    s32 = auto()
    u32 = auto()
    s64 = auto()
    u64 = auto()

    f32 = auto()
    f64 = auto()

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
    TypeKind.s8,
    TypeKind.u8,
    TypeKind.s16,
    TypeKind.u16,
    TypeKind.s32,
    TypeKind.u32,
    TypeKind.s64,
    TypeKind.u64,
])

float_types = frozenset([
    TypeKind.f32,
    TypeKind.f64,
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
            case TypeKind.s8 | TypeKind.u8 | TypeKind.s16 | TypeKind.u16 | TypeKind.s32 | TypeKind.u32 | TypeKind.s64 | TypeKind.u64:
                return self.kind == other.kind or other.kind == TypeKind.literal_int
            case TypeKind.f32 | TypeKind.f64:
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
                raise ValueError("statics aren't ready yet")

            case TypeKind.alias:
                # TODO: Generic aliases. Need to map potentially new generic params around and then reduce the type if required
                raise ValueError("aliases aren't ready yet")
            
            case TypeKind.distinct:
                # TODO: distinct generics are tough
                raise ValueError("distincts aren't ready yet")
            

            # user types

            case TypeKind.generic_inst:
                raise ValueError("generic instances aren't ready yet")

            case TypeKind.generic_type:
                raise ValueError("generic types aren't ready yet")

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
                raise ValueError(f"Unimplemented type comparison: {self.kind}")

    def instantiate_literal(self, graph: ModuleGraph) -> Type:
        """
        Turns a literal into a concrete type
        """
        assert self.kind in literal_types
        match self.kind:
            case TypeKind.literal_int:
                return graph.builtin_scope.lookup_type("u32")
            case TypeKind.literal_float:
                return graph.builtin_scope.lookup_type("f32")
            case TypeKind.literal_bool:
                return graph.builtin_scope.lookup_type("bool")
            case TypeKind.literal_string:
                return graph.builtin_scope.lookup_type("string")
            case _:
                raise ValueError(f"Forgot a literal type: {self.kind}")

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
            raise NotImplementedError(self)


class Scope:
    def __init__(self, graph: ModuleGraph) -> None:
        self._local_syms: list[Symbol] = []
        self.parent: Scope = None
        self.module_graph = graph # TODO: Get rid of builtin hack

    def iter_function_defs(self, name: str) -> Iterator[Symbol]:
        # prefer magics
        if self.module_graph.builtin_scope != self:
            for sym in self.module_graph.builtin_scope.iter_function_defs(name):
                if sym.name == name:
                    yield sym

        for sym in self._local_syms:
            # TODO: Maybe split functions and types here? Also handle builtin types here, no reason they should be different
            if sym.type.kind not in callable_types:
                continue
            if sym.name == name:
                yield sym
        if self.parent != None:
            yield from self.parent.iter_function_defs(name)

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

    def put(self, name: str, checked=True, shallow=False) -> Symbol:
        assert type(name) == str
        if checked and self.lookup(name, shallow=shallow): raise ValueError(f"redefinition of {name}")

        if name in RESERVED_KEYWORDS:
            raise ValueError(f"Cannot use reserved keyword `{name}` as a symbol name")

        result = Symbol(self.module_graph.sym_id_gen.next(), name=name)
        self._local_syms.append(result)
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
        if self.lookup(name, shallow=True): raise ValueError(f"redefinition of magic sym: {name}")
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


class CompCtx(lark.visitors.Interpreter):
    def __init__(self, module: Module, graph: ModuleGraph) -> None:
        self.module = module
        self.graph = graph
        self.current_scope = self.module.global_scope
        self.fn_ret_type_stack: list[Type] = []
        self.in_loop_counter = 0

    def open_scope(self):
        self.current_scope = self.current_scope.make_child()

    def close_scope(self):
        self.current_scope = self.current_scope.parent

    def get_infer_type(self) -> Type:
        return Type(kind=TypeKind.infer, sym=None)

    def get_unit_type(self) -> Type:
        return self.current_scope.lookup_type("unit", shallow=True)

    # overrides
    def visit(self, tree: Tree, expected_type: Type) -> Node:
        return self._visit_tree(tree, expected_type)

    def _visit_tree(self, tree: Tree, expected_type: Type):
        f = getattr(self, tree.data)
        wrapper = getattr(f, 'visit_wrapper', None)
        if wrapper is not None:
            raise NotImplementedError()
            #return f.visit_wrapper(f, tree.data, tree.children, tree.meta)
        else:
            return f(tree, expected_type)
    
    def visit_children(self, tree: Tree, expected_type: Type) -> list:
        return [self._visit_tree(child, expected_type) if isinstance(child, Tree) else child
                for child in tree.children]

    def __default__(self, tree, expected_type: Type):
        raise ValueError(f"{tree=}")
        return self.visit_children(tree, expected_type)

    # new functions
    def statement(self, tree: Tree, expected_type: Type):
        assert len(tree.children) == 1, f"{len(tree.children)} --- {tree.children=}"
        return self.visit(tree.children[0], expected_type)


    def handle_break_or_continue(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit
        assert len(tree.children) == 0
        if self.in_loop_counter < 1:
            raise ValueError("Break or continue outside of a loop")
        if tree.data == "break_stmt":
            return NodeBreak(self.get_unit_type())
        elif tree.data == "continue_stmt":
            return NodeContinue(self.get_unit_type())
        else:
            raise ValueError(f"Somehow a bad break or continue has been given: {tree.data}")
        

    def break_stmt(self, tree: Tree, expected_type: Type):
        return self.handle_break_or_continue(tree, expected_type)

    def continue_stmt(self, tree: Tree, expected_type: Type):
        return self.handle_break_or_continue(tree, expected_type)


    def while_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit
        # This has to use the outer scope, so a new scope is only opened once this has been checked in full
        while_cond = self.visit(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert while_cond.type.kind == TypeKind.bool

        # block_stmt opens a scope
        self.in_loop_counter += 1 # needed to check if break and continue are valid
        body = self.visit(tree.children[1], self.get_unit_type())
        self.in_loop_counter -= 1
        assert isinstance(body, NodeBlockStmt)

        return NodeWhileStmt(while_cond, body, self.get_unit_type())

    def if_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit # TODO: if expressions are not unit, need to guarantee valid else branch
        # Same scoping story as in while_stmt
        if_cond = self.visit(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert if_cond.type.kind == TypeKind.bool
        if_body = self.visit(tree.children[1], self.get_unit_type())
        assert isinstance(if_body, NodeBlockStmt)

        else_body = None
        if tree.children[2] != None:
            else_body = self.visit(tree.children[2], self.get_unit_type())
            assert isinstance(else_body, NodeBlockStmt)
        else:
            # Always inject an empty else case if none is provided
            else_body = NodeBlockStmt(self.get_unit_type())
        return NodeIfStmt(if_cond, if_body, else_body, self.get_unit_type()) # TODO: Pass along branch types once they exist


    def block_stmt(self, tree: Tree, expected_type: Type):
        self.open_scope()
        if len(tree.children) == 0:
            return NodeBlockStmt(self.get_unit_type())

        # Assume unit type if nothing is expected, fixed later
        # Have to be very careful with symbols, we do not want to use one that only exists later
        result = NodeBlockStmt(expected_type if expected_type != None else self.get_unit_type())

        (tree_children, last_child) = (tree.children[0:len(tree.children)-1], tree.children[-1])
        for child in tree_children:
            # All but the last have to be unit typed
            result.add(self.visit(child, self.get_unit_type()))

        if expected_type != None:
            result.add(self.visit(last_child, expected_type))
            assert expected_type.types_compatible(result.children[-1].type), f"Expected type {expected_type.sym.render()} for block but got {result.children[-1].type.sym.render()}"
            result.type = result.children[-1].type
        elif len(result.children) > 0:
            result.add(self.visit(last_child, None))
            result.type = result.children[-1].type

        self.close_scope()
        return result
    
    def block_expression(self, tree: Tree, expected_type: Type):
        return self.block_stmt(tree, expected_type)

    def grouped_expression(self, tree: Tree, expected_type: Type):
        inner = self.visit(tree.children[0], expected_type)
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
                    raise RuntimeError("Unsupported empty literal")

        if expected_type != None:
            if expected_type.kind in free_infer_types:
                expected_type = self.current_scope.lookup_type(lookup_name, shallow=True)
            else:
                assert expected_type.types_compatible(Type(literal_kind, sym=None))
            return node_type(value=value, type=expected_type)
        else:
            return node_type(value=value, type=Type(literal_kind, sym=None))

    def integer(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "u32", TypeKind.literal_int, NodeIntLit, int)
    
    def decimal(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "f32", TypeKind.literal_float, NodeFloatLit, float)
    
    def bool(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "bool", TypeKind.literal_bool, NodeBoolLit, lambda x: x == "true")
    
    def string(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "string", TypeKind.literal_string, NodeStringLit, str)


    def binary_expression(self, tree: Tree, expected_type: Type):
        lhs = self.visit(tree.children[0], None)
        # TODO: Can the symbol be captured somehow?
        op = tree.children[1].data.value
        # a dot expression does not work by the same type rules
        if op != "dot":
            rhs = self.visit(tree.children[2], None)

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
                lhs = self.visit(tree.children[0], rhs.type)
            elif rhs.type.kind in literal_types and lhs.type.kind not in literal_types:
                rhs = self.visit(tree.children[2], lhs.type)
            
            # both literal
            else:
                # if possible we ask for help from expected_type
                if expected_type != None and expr_type.types_compatible(expected_type):
                    lhs = self.visit(tree.children[0], expected_type)
                    rhs = self.visit(tree.children[2], expected_type)
                    assert lhs.type.types_compatible(rhs.type)
                    expr_type = lhs.type

                # no expectation, let them infer their own types
                elif expected_type == None:
                    lhs = self.visit(tree.children[0], self.get_infer_type())
                    rhs = self.visit(tree.children[2], self.get_infer_type())
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
                lhs = self.visit(tree.children[0], None)
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
                raise ValueError(f"Unimplemented binary op: {op}")


    def identifier(self, tree: Tree, expected_type: Type):
        val = tree.children[0].value
        sym = self.current_scope.lookup(val)
        if sym:
            if expected_type != None:
                assert sym.type.types_compatible(expected_type)
            return NodeSymbol(sym, type=sym.type)
        # TODO: Error reporting
        raise ValueError(f"Bad identifier: {val}")

    def return_user_type(self, tree: Tree, expected_type: Type):
        return self.user_type(tree, expected_type)

    def user_type(self, tree: Tree, expected_type: Type):
        return self.visit(tree.children[0], None) # TODO: Enforce `type` metatype

    def assignment(self, tree: Tree, expected_type: Type):
        # TODO: Assignments could technically act as expressions too
        assert expected_type.kind == TypeKind.unit

        lhs = self.visit(tree.children[0], None)
        rhs = self.visit(tree.children[1], lhs.type)
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
                raise NotImplementedError
        
        return NodeAssignment(lhs, rhs, self.get_unit_type())

    def let_stmt(self, tree: Tree, expected_type: Type):
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
            type_sym = self.visit(type_tree, None)
            assert isinstance(type_sym, NodeSymbol)
            f += 1
        
        val_node_expected_type = type_sym.type if type_sym != None else self.get_infer_type()
        val_node = self.visit(tree.children[f], val_node_expected_type)
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

    def return_stmt(self, tree: Tree, expected_type: Type):
        assert len(self.fn_ret_type_stack) > 0, "Return outside of a function"
        # TODO: Make sure this passes type checks
        expr = None
        if tree.children[0] != None:
            expr = self.visit(tree.children[0], self.fn_ret_type_stack[-1])
        else:
            expr = NodeEmpty(self.fn_ret_type_stack[-1])        
        assert self.fn_ret_type_stack[-1].types_compatible(expr.type), f"Incompatible return({expr.type.sym.render()}) for function type({self.fn_ret_type_stack[-1].render()})"
        return NodeReturn(expr=expr, type=self.get_unit_type())

    def alias_definition(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit

        src = self.visit(tree.children[1], None)
        assert isinstance(src, NodeSymbol)

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

    def struct_field(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit

        ident = tree.children[0].children[0].value
        sym = self.current_scope.put_let(ident, checked=True, shallow=True)

        typ_sym_node = self.visit(tree.children[1], None)
        assert isinstance(typ_sym_node, NodeSymbol) # TODO: Assert that this is actually a type sym. Current system isn't ready yet

        sym.type = typ_sym_node.type

        return NodeStructField(sym, typ_sym_node.symbol.type)

    def struct_definition(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit
        ident = tree.children[0].children[0].value
        sym = self.current_scope.put_type(ident)

        fields = []
        if tree.children[1] != None:
            self.open_scope()
            for field_node in tree.children[1:]:
                # TODO: Recursion check. Recursion is currently unrestricted and permits infinite recursion
                #   For proper recursion handling we also gotta ensure we don't try to access a type that is currently being processed like will be the case with mutual recursion
                field = self.visit(field_node, self.get_unit_type())
                fields.append(field)
            self.close_scope()

        def_node = NodeStructDefinition(sym, fields, self.get_unit_type())
        sym.definition_node = def_node
        return def_node

    def fn_definition_args(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit

        params = []

        if len(tree.children) == 1 and tree.children[0] is None:
            return NodeFnParameters(params)

        for child in tree.children:
            assert child.data == "fn_definition_arg"
            # TODO: Mutable args
            ident = child.children[0].children[0].value
            sym = self.current_scope.put_let(ident, shallow=True) # effectively a let that permits shallow shadowing
            type_node = self.visit(child.children[1], None)
            sym.type = type_node.symbol.type # TODO: This is not clean at all
            params.append((NodeSymbol(sym, sym.type), type_node))

        return NodeFnParameters(params)

    def fn_definition(self, tree: Tree, expected_type: Type):
        assert expected_type.kind == TypeKind.unit

        ident_node = tree.children[0]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value

        # must open a scope here to isolate the params
        self.open_scope()
        
        args_node = self.visit(tree.children[1], self.get_unit_type())
        assert isinstance(args_node, NodeFnParameters)
        ret_type_node = NodeSymbol(self.get_unit_type().sym, self.get_unit_type())
        ret_type = ret_type_node.type
        if tree.children[2] != None:
            ret_type_node = self.visit(tree.children[2], None)
            assert isinstance(ret_type_node, NodeSymbol)
            ret_type = ret_type_node.type # TODO: Fix this nonsense, type should be `type`, not whatever the type evaluates to

        # TODO: Use a type cache to load function with the same type from it for easier matching
        fn_type = Type(
            kind=TypeKind.function,
            sym=None,
            data=([x[1].type for x in args_node.args], ret_type_node)
        )

        # The sym must be created here to make recursive calls work without polluting the arg scope
        sym = self.current_scope.parent.put_function(ident, fn_type)
        fn_type.sym = sym

        # TODO: Make this work for generics later
        self.fn_ret_type_stack.append(ret_type)

        body_node: NodeBlockStmt = self.visit(tree.children[3], self.get_infer_type())
        assert isinstance(body_node, NodeBlockStmt)

        # TODO: Simplify checks, we can rely on the fact that it has to be transformed into `return x`
        if len(body_node.children) > 0:
            last_body_node = body_node.children[-1]
            if last_body_node.type.kind == TypeKind.unit and not expected_type.kind == TypeKind.unit and not isinstance(last_body_node, NodeReturn):
                assert False, f"Returning a unit type for expected type {ret_type.sym.render()} is not permitted"
            elif isinstance(last_body_node, NodeReturn):
                pass # return is already checked
            elif last_body_node.type.kind in literal_types:
                body_node = self.visit(tree.children[3], ret_type)
                last_body_node = body_node.children[-1]
                if last_body_node.type.kind in builtin_userspace_types:
                    assert last_body_node.type.types_compatible(ret_type), f"Invalid return expression type {last_body_node.type.sym.render()} for return type {ret_type.sym.render()}"
                body_node.children[-1] = NodeReturn(last_body_node, self.get_unit_type())
            else:
                assert last_body_node.type.types_compatible(ret_type), f"Invalid return expression type {last_body_node.type.sym.render()} for return type {ret_type.sym.render()}"
                if last_body_node.type.kind == TypeKind.unit:
                    body_node.children.append(NodeReturn(NodeEmpty(self.get_unit_type()), self.get_unit_type()))
                else:
                    body_node.children[-1] = NodeReturn(last_body_node, self.get_unit_type())
        else:
            body_node.children.append(NodeReturn(NodeEmpty(self.get_unit_type()), self.get_unit_type()))

        self.fn_ret_type_stack.pop()

        self.close_scope()

        sym.type = fn_type

        res = NodeFnDefinition(sym, args_node, body_node, self.get_unit_type())
        sym.definition_node = res
        return res

    def fn_call_expr(self, tree: Tree, expected_type: Type):
        # TODO: Once overloads/methods are in, this must scan visible symbols and retrieve a list. Handled in identifier though
        unresolved_args: list[Node] = []
        if tree.children[1] != None:
            # child 0 is None in empty calls i.e f()
            if tree.children[1].children[0] is not None:
                for i in range(0, len(tree.children[1].children)):
                    unresolved_args.append(self.visit(tree.children[1].children[i], self.get_infer_type()))
        
        callee = None
        if len(tree.children[0].children) > 0 and tree.children[0].children[0].data == "identifier":
            # TODO: Better handling, use a candidate node instead
            for fn in self.current_scope.iter_function_defs(tree.children[0].children[0].children[0].value):
                if fn.type.kind == TypeKind.type:
                    assert len(unresolved_args) == 0, "Calling a struct with arguments is not yet supported" # TODO 
                    callee = NodeSymbol(fn, fn.type)
                    break
                else:
                    if len(unresolved_args) != len(fn.type.function_arg_types()):
                        continue
                    matches = True
                    for i in range(0, len(unresolved_args)):
                        if not unresolved_args[i].type.types_compatible(fn.type.function_arg_types()[i]):
                            matches = False
                            break
                    if matches:
                        callee = NodeSymbol(fn, fn.type)
                        break
        else:
            callee = self.visit(tree.children[0], None)
        # TODO: allow non-symbol calls i.e. function pointers
        assert callee != None, f"No matching overload found for {tree.children[0].children[0].children[0]}"
        assert isinstance(callee, NodeSymbol), "can only call symbols"

        params = []
        ret_type = None
        if callee.type.kind == TypeKind.type:
            ret_type = callee.type
        else:
            arg_types = callee.symbol.type.data[0]
            ret_type_node = callee.symbol.type.data[1]
            ret_type = None
            if ret_type_node != None:
                assert isinstance(ret_type_node, NodeSymbol), f"{ret_type_node=}"
                ret_type = ret_type_node.type # TODO: Fix garbage type, should be a `type`, not evaluated type
            else:
                ret_type = self.current_scope.lookup_type("unit", shallow=True)

            params = []
            if tree.children[1] != None:
                # child 0 is None in empty calls i.e f()
                if tree.children[1].children[0] is not None:
                    assert len(tree.children[1].children) == len(arg_types)
                    for i in range(0, len(arg_types)):
                        params.append(self.visit(tree.children[1].children[i], arg_types[i]))
            else:
                assert len(arg_types) == 0

        return NodeFnCall(
            callee=callee,
            args=params,
            type=ret_type
        )

    def expression(self, tree: Tree, expected_type: Type):
        # TODO: Handle longer expressions?
        assert len(tree.children) == 1
        result = []
        tree: Tree = tree
        for child in tree.children:
            result.append(self.visit(child, expected_type))
        if len(result) == 1:
            return result[0]
        else:
            assert False

    def start(self, tree: Tree, expected_type: Type):
        result = NodeStmtList(self.current_scope.lookup_type("unit", shallow=True))
        for child in tree.children:
            node = self.visit(child, self.get_unit_type())
            assert isinstance(node, Node)
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

        self.builtin_scope.put_builtin_type(TypeKind.s8)
        self.builtin_scope.put_builtin_type(TypeKind.u8)
        self.builtin_scope.put_builtin_type(TypeKind.s16)
        self.builtin_scope.put_builtin_type(TypeKind.u16)
        self.builtin_scope.put_builtin_type(TypeKind.s32)
        self.builtin_scope.put_builtin_type(TypeKind.u32)
        self.builtin_scope.put_builtin_type(TypeKind.s64)
        self.builtin_scope.put_builtin_type(TypeKind.u64)

        self.builtin_scope.put_builtin_type(TypeKind.f32)
        self.builtin_scope.put_builtin_type(TypeKind.f64)

        self.builtin_scope.put_builtin_type(TypeKind.pointer)

        # TODO
        self.builtin_scope.put_builtin_type(TypeKind.string)
        self.builtin_scope.put_builtin_type(TypeKind.array)
        self.builtin_scope.put_builtin_type(TypeKind.static)


        # TODO: hack
        magic_sym = Symbol("-1", "magic")
        magic_type = Type(TypeKind.magic, magic_sym)
        magic_sym.type = magic_type

        dbg_sym_type = Type(TypeKind.function, None, ([magic_type], NodeSymbol(unit_type_sym, unit_type_sym.type)))
        dbg_sym = self.builtin_scope.put_magic_function("dbg", dbg_sym_type)
        dbg_sym_type.sym = dbg_sym


    def load(self, name: str, file_contents: str) -> Module:
        assert name not in self.modules
        mod = Module(name, self._next_id, file_contents, self)
        self._next_id += 1
        self.modules[name] = mod
        # TODO: Make sure the module isn't already being processed
        ast: NodeStmtList = CompCtx(mod, self).visit(mod.lark_tree, None)
        mod.ast = ast
        return mod
