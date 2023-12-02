from __future__ import annotations

from typing import Any, Optional
from enum import Enum, auto

import hashlib
import textwrap

import lark.visitors
from lark import Token, Tree

from serqlane.parser import SerqParser


class Node:
    def __init__(self, type: Type | None = None) -> None:
        self.type = type

    def render(self) -> str:
        raise NotImplementedError(f"{type(self)}")

class NodeSymbol(Node):
    def __init__(self, symbol: Symbol, type: Type | None = None) -> None:
        super().__init__(type)
        self.symbol = symbol

    def render(self) -> str:
        # TODO: Unique global identifier later
        return f"{self.symbol.render()}"

class NodeStmtList(Node):
    def __init__(self) -> None:
        super().__init__()
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
    def __init__(self, sym_node: NodeSymbol, expr: Node):
        super().__init__()
        self.sym_node = sym_node
        self.expr = expr

    def render(self) -> str:
        is_mut = self.sym_node.symbol.mutable
        return f"let {"mut " if is_mut else ""}{self.sym_node.render()}{": " + self.sym_node.type.render()} = {self.expr.render()};"

class NodeAssignment(Node):
    def __init__(self, lhs: Node, rhs: Node) -> None:
        super().__init__(None)
        self.lhs = lhs
        self.rhs = rhs

    def render(self) -> str:
        return f"{self.lhs.render()} = {self.rhs.render()};"


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

class NodeDotExpr(NodeBinaryExpr):
    def render(self) -> str:
        return f"({self.lhs.render()}.{self.rhs.render()})"

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


# others
class NodeBreak(Node):
    def __init__(self) -> None:
        super().__init__(None)

    def render(self) -> str:
        return "break;"

class NodeContinue(Node):
    def __init__(self) -> None:
        super().__init__(None)

    def render(self) -> str:
        return "continue;"


class NodeBlockStmt(NodeStmtList):
    def __init__(self, scope: Scope) -> None:
        super().__init__()
        self.scope = scope # TODO: Should this actually be stored?

    def render(self) -> str:
        inner = super().render()
        return f"{{\n{textwrap.indent(inner, "  ")}\n}}"

class NodeWhileStmt(Node):
    def __init__(self, cond_expr: Node, body: NodeBlockStmt) -> None:
        super().__init__(None)
        self.cond_expr = cond_expr
        self.body = body
    
    def render(self) -> str:
        cond = self.cond_expr.render()
        body = self.body.render()
        return f"while ({cond}) {body}"

class NodeIfStmt(Node):
    def __init__(self, cond_expr: Node, if_body: NodeBlockStmt, else_body: NodeBlockStmt) -> None:
        super().__init__(None)
        self.cond_expr = cond_expr
        self.if_body = if_body
        self.else_body = else_body

    def render(self) -> str:
        cond = self.cond_expr.render()
        if_body = self.if_body.render()
        else_body = self.else_body.render()
        return f"if ({cond}) {if_body} else {else_body}"


class NodeReturn(Node):
    def __init__(self, expr: Node) -> None:
        super().__init__(None)
        self.expr = expr

    def render(self) -> str:
        return f"return {self.expr.render()};"

class NodeFnParameters(Node):
    def __init__(self, args: list[tuple[NodeSymbol, Node]]):
        """
        args is a list of (param_sym, type_node)
        type is stored as a node to help with generic instantiation later
        """
        self.args = args

    def render(self) -> str:
        return ", ".join([x[0].render() + ": " + x[1].type.render() for x in self.args])

class NodeFnDefinition(Node):
    def __init__(self, sym: Symbol, params: NodeFnParameters, body: NodeBlockStmt) -> None:
        super().__init__(None)
        assert sym.type.kind == TypeKind.function
        self.sym = sym # holds all of the actual type info
        self.sym.definition_node = self # store a reference to self so we can resolve named args
        self.params = params # Need this for named args like foo(a=10)
        self.body = body

    def render(self) -> str:
        return f"fn {self.sym.render()}({self.params.render()}) {self.body.render()}"

class NodeFnCall(Node):
    def __init__(self, callee: Node, args: list[Node], type: Type | None = None) -> None:
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
    def __init__(self, id: str, name: str, type: Type | None = None, mutable: bool = False) -> None:
        # TODO: Should store the source node, symbol kinds
        self.id = id
        self.name = name
        self.type = type
        self.exported = False
        self.mutable = mutable
        self.definition_node: Node = None

    def render(self) -> str:
        # TODO: Use type info to render generics and others
        return f"{self.name}_{self.id}"


class TypeKind(Enum):
    error = auto() # bad type
    infer = auto() # marker type to let nodes know to infer their own types
    
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


# TODO: Add the other appropriate types
builtin_userspace_types = frozenset(list(int_types) + list(float_types) + [TypeKind.bool, TypeKind.char, TypeKind.string])

class Type:
    def __init__(self, kind: TypeKind, sym: Symbol, data: Any = None) -> None:
        self.kind = kind
        self.data = data # TODO: arbitrary data for now
        self.sym = sym
        # TODO: Add a type id later

    # TODO: Turn into type_relation. literal<->concrete means uninstantiated_literal
    def types_compatible(self, other: Type) -> bool:
        """
        other is always the target
        """
        if self.kind == TypeKind.infer or other.kind == TypeKind.infer:
            return True

        # TODO: Match variant, like generic inst of generic type
        match self.kind:
            case TypeKind.error:
                return False
            
            # TODO: Not sure what to do about these
            case TypeKind.unit:
                raise ValueError("units aren't ready yet")
            
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
                if id(self) == id(other):
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
                return graph.builtin_scope.lookup_type("int")
            case TypeKind.literal_float:
                return graph.builtin_scope.lookup_type("float")
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
        if self.kind == TypeKind.function:
            args = ", ".join([x.render() for x in self.data[0]])
            return f"fn({args}): {self.data[1].render()}"
        else:
            raise NotImplementedError()


class Scope:
    def __init__(self, graph: ModuleGraph) -> None:
        self._local_syms: list[Symbol] = []
        self.parent: Scope = None
        self.module_graph = graph # TODO: Get rid of builtin hack

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

    def put(self, name: str, checked=True) -> Symbol:
        assert type(name) == str
        if checked and self.lookup(name): raise ValueError(f"redefinition of {name}")
        
        result = Symbol(self.module_graph.sym_id_gen.next(), name=name)
        self._local_syms.append(result)
        return result

    def put_builtin_type(self, kind: TypeKind) -> Symbol:
        # TODO: Get rid of this hack
        sym = self.put(kind.name, checked=False)
        sym.type = Type(kind=kind, sym=sym)
        return sym

    def put_let(self, name: str, mutable=False) -> Symbol:
        sym = self.put(name)
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

    def get_infer_type(self) -> Type:
        return Type(kind=TypeKind.infer, sym=None)

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
        assert expected_type == None
        assert len(tree.children) == 0
        if self.in_loop_counter < 1:
            raise ValueError("Break or continue outside of a loop")
        if tree.data == "break_stmt":
            return NodeBreak()
        elif tree.data == "continue_stmt":
            return NodeContinue()
        else:
            raise ValueError(f"Somehow a bad break or continue has been given: {tree.data}")
        

    def break_stmt(self, tree: Tree, expected_type: Type):
        return self.handle_break_or_continue(tree, expected_type)

    def continue_stmt(self, tree: Tree, expected_type: Type):
        return self.handle_break_or_continue(tree, expected_type)


    def while_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type == None
        # This has to use the outer scope, so a new scope is only opened once this has been checked in full
        while_cond = self.visit(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert while_cond.type.kind == TypeKind.bool

        # block_stmt opens a scope
        self.in_loop_counter += 1 # needed to check if break and continue are valid
        body = self.visit(tree.children[1], None)
        self.in_loop_counter -= 1
        assert isinstance(body, NodeBlockStmt)

        return NodeWhileStmt(while_cond, body)

    def if_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type == None
        # Same scoping story as in while_stmt
        if_cond = self.visit(tree.children[0], self.current_scope.lookup_type("bool", shallow=True))
        assert if_cond.type.kind == TypeKind.bool
        if_body = self.visit(tree.children[1], None)
        assert isinstance(if_body, NodeBlockStmt)

        else_body = None
        if tree.children[2] != None:
            else_body = self.visit(tree.children[2], None)
            assert isinstance(else_body, NodeBlockStmt)
        else:
            # Always inject an empty else case if none is provided
            else_body = NodeBlockStmt(self.current_scope.make_child())
        return NodeIfStmt(if_cond, if_body, else_body)


    def block_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type == None
        self.current_scope = self.current_scope.make_child()
        result = NodeBlockStmt(self.current_scope)
        # Have to be very careful with symbols, we do not want to use one that only exists later
        for child in tree.children:
            result.add(self.visit(child, None))
        self.current_scope = self.current_scope.parent
        return result

    def grouped_expression(self, tree: Tree, expected_type: Type):
        inner = self.visit(tree.children[0], expected_type)
        return NodeGrouped(inner, inner.type)


    def handle_literal(self, tree: Tree, expected_type: Type, lookup_name: str, literal_kind: TypeKind, node_type: Type[Node], conv_fn):
        val = tree.children[0].value
        if expected_type != None:
            if expected_type.kind == TypeKind.infer:
                expected_type = self.current_scope.lookup_type(lookup_name, shallow=True)
            else:
                assert expected_type.types_compatible(Type(literal_kind, sym=None))
            return node_type(value=conv_fn(val), type=expected_type)
        else:
            return node_type(value=conv_fn(val), type=Type(literal_kind, sym=None))

    def integer(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "int", TypeKind.literal_int, NodeIntLit, int)
    
    def decimal(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "float", TypeKind.literal_float, NodeFloatLit, float)
    
    def bool(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "bool", TypeKind.literal_bool, NodeBoolLit, lambda x: x == "true")
    
    def string(self, tree: Tree, expected_type: Type):
        return self.handle_literal(tree, expected_type, "string", TypeKind.literal_string, NodeStringLit, str)


    def binary_expression(self, tree: Tree, expected_type: Type):
        lhs = self.visit(tree.children[0], None)
        # TODO: Can the symbol be captured somehow?
        op = tree.children[1].data.value
        rhs = self.visit(tree.children[2], None)

        # TODO: Dot expr
        if not lhs.type.types_compatible(rhs.type):
            # TODO: Error reporting
            raise ValueError(f"Incompatible values in binary expression: `{lhs.render()}`:{lhs.type.render()} {op} `{rhs.render()}`:{rhs.type.render()}")
        
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
                raise ValueError("Dot expressions are not ready")
            case _:
                raise ValueError(f"Unimplemented binary op: {op}")


    def identifier(self, tree: Tree, expected_type: Type):
        val = tree.children[0].value
        sym = self.current_scope.lookup(val)
        if sym:
            if expected_type != None:
                sym.type.types_compatible(expected_type)
            return NodeSymbol(sym, type=sym.type)
        # TODO: Error reporting
        raise ValueError(f"Bad identifier: {val}")

    def user_type(self, tree: Tree, expected_type: Type):
        return self.visit(tree.children[0], None) # TODO: Enforce `type` metatype

    def assignment(self, tree: Tree, expected_type: Type):
        # TODO: Assignments could technically act as expressions too
        assert expected_type == None

        lhs = self.visit(tree.children[0], None)
        rhs = self.visit(tree.children[1], lhs.type)
        assert lhs.type.types_compatible(rhs.type)

        match lhs:
            case NodeSymbol():
                if not lhs.symbol.mutable:
                    raise ValueError(f"{lhs.symbol.name} is not mutable")
            case _:
                raise NotImplementedError
        
        return NodeAssignment(lhs, rhs)

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

        resolved_type = None
        if type_sym != None:
            resolved_type = type_sym.type
            # check types for compatibility
            if not val_node.type.types_compatible(resolved_type):
                # TODO: Error reporting
                raise ValueError(f"Variable type {type_sym.symbol.name} is not compatible with value of type {val_node.type.render()}")
            else:
                assert val_node.type.kind not in literal_types
            # no need to resem the node, it should already be with the current type resolution
        else:
            # infer type from value
            # TODO: Instantiate types, for now only literals            
            if val_node.type.kind in builtin_userspace_types:
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
            expr=val_node
        )

    def return_stmt(self, tree: Tree, expected_type: Type):
        assert expected_type == None
        assert len(self.fn_ret_type_stack) > 0, "Return outside of a function"
        # TODO: Make sure this passes type checks
        return NodeReturn(expr=self.visit(tree.children[0], self.fn_ret_type_stack[-1]))

    def fn_definition_args(self, tree: Tree, expected_type: Type):
        assert expected_type == None

        params = []
        for child in tree.children:
            assert child.data == "fn_definition_arg"
            # TODO: Mutable args
            ident = child.children[0].children[0].value
            sym = self.current_scope.put_let(ident) # effectively a let
            type_node = self.visit(child.children[1], None)
            sym.type = type_node.symbol.type # TODO: This is not clean at all
            params.append((sym, type_node))

        return NodeFnParameters(params)

    def fn_definition(self, tree: Tree, expected_type: Type):
        assert expected_type == None

        ident_node = tree.children[0]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value


        # must open a scope here to isolate the params
        fn_scope = self.current_scope.make_child()
        self.current_scope = fn_scope
        
        args_node = self.visit(tree.children[1], None)
        assert isinstance(args_node, NodeFnParameters)
        ret_type_node = self.visit(tree.children[2], None)
        assert isinstance(ret_type_node, NodeSymbol)

        # The sym must be created here to make recursive calls work without polluting the arg scope
        sym = self.current_scope.parent.put(ident)

        # TODO: Make this work for generics later
        self.fn_ret_type_stack.append(ret_type_node.type)

        # TODO: Use a type cache to load function with the same type from it for easier matching
        fn_type = Type(
            kind=TypeKind.function,
            sym=None,
            data=([x[1].type for x in args_node.args], ret_type_node)
        )

        body_node = self.visit(tree.children[3], None) # TODO: once block expressions work, this should expect the return type
        assert isinstance(body_node, NodeBlockStmt)

        self.fn_ret_type_stack.pop()

        # restore scope
        self.current_scope = fn_scope.parent

        sym.type = fn_type

        res = NodeFnDefinition(sym, args_node, body_node)
        sym.definition_node = res
        return res

    def fn_call_expr(self, tree: Tree, expected_type: Type):
        # TODO: Once overloads/methods are in, this must scan visible symbols and retrieve a list. Handled in identifier though
        callee = self.visit(tree.children[0], None)
        arg_types = callee.symbol.type.data[0]
        ret_type_node = callee.symbol.type.data[1]
        assert isinstance(ret_type_node, NodeSymbol)

        params = []
        if tree.children[1] != None:
            assert len(tree.children[1].children) == len(arg_types)
            for i in range(0, len(arg_types)):
                params.append(self.visit(tree.children[1].children[i], arg_types[i]))
        else:
            assert len(arg_types) == 0

        return NodeFnCall(
            callee=callee,
            args=params,
            type=ret_type_node.symbol.type
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
        result = NodeStmtList()
        for child in tree.children:
            node = self.visit(child, None) # TODO: Force None for now
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

        # TODO: hack
        native_int = self.builtin_scope.put("int", checked=False)
        native_int.type = self.builtin_scope.lookup(TypeKind.int64.name).type

        native_uint = self.builtin_scope.put("uint", checked=False)
        native_uint.type = self.builtin_scope.lookup(TypeKind.uint64.name).type

        native_float = self.builtin_scope.put("float", checked=False)
        native_float.type = self.builtin_scope.lookup(TypeKind.float64.name).type
        
        # TODO
        self.builtin_scope.put_builtin_type(TypeKind.string)
        self.builtin_scope.put_builtin_type(TypeKind.array)
        self.builtin_scope.put_builtin_type(TypeKind.static)


    def load(self, name: str, file_contents: str) -> Module:
        assert name not in self.modules
        mod = Module(name, self._next_id, file_contents, self)
        self._next_id += 1
        self.modules[name] = mod
        
        # TODO: Make sure the module isn't already being processed
        ast: NodeStmtList = CompCtx(mod, self).visit(mod.lark_tree, None)
        mod.ast = ast
        return mod
