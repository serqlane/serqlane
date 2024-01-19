from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional, Iterator

import pathlib
import hashlib
import textwrap
import warnings

from serqlane.parser import Token, Tree, SerqParser
from serqlane.common import SerqInternalError


DEBUG = False

MAGIC_MODULE_NAME = "serqlib/magics"


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


class SerqTypeInferError(Exception): ... # TODO: Get rid of this


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

class NodeModuleSymbol(Node):
    def __init__(self, symbol: Symbol, type: Type) -> None:
        super().__init__(type)
        self.symbol = symbol

    def render(self) -> str:
        return f"{self.symbol.render()}"

class NodeRangeExpr(Node):
    def __init__(self, start: Node, stop: Node, type: Type) -> None:
        super().__init__(type)
        self.start = start
        self.stop = stop

    def render(self) -> str:
        return f"{self.start.render()} .. {self.stop.render()}"

class NodeStmtList(Node):
    def __init__(self, type: Type, children: Optional[list[Node]] = None) -> None:
        super().__init__(type)
        self.children: list[Node] = [] if children == None else children

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
        return f"\"{self.value.encode("unicode_escape").decode("utf-8")}\""

class NodeCharLit(NodeLiteral[str]):
    def render(self) -> str:
        return f"'{self.value.encode("unicode_escape").decode("utf-8")}'"

class NodeLet(Node):
    def __init__(self, sym_node: NodeSymbol, expr: Node, type: Type):
        super().__init__(type)
        self.sym_node = sym_node
        self.expr = expr

    def render(self) -> str:
        is_mut = self.sym_node.symbol.mutable or self.sym_node.symbol.hidden_mutable
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
    def __init__(self, lhs: Node, rhs: NodeSymbol) -> None:
        super().__init__(rhs.type)
        self.lhs = lhs
        self.rhs = rhs
        assert isinstance(self.rhs, NodeSymbol)

    def render(self) -> str:
        return f"{self.lhs.render()}.{self.rhs.render()}"


class NodeUnaryExpression(Node):
    def __init__(self, expr: Node, type: Type):
        super().__init__(type)
        self.expr = expr

class NodeNotExpression(NodeUnaryExpression):
    def render(self) -> str:
        return f"not ({self.expr.render()})"

class NodeNegExpression(NodeUnaryExpression):
    def render(self) -> str:
        return f"-({self.expr.render()})"


# others
class NodeBreak(Node):
    def render(self) -> str:
        return "break"

class NodeContinue(Node):
    def render(self) -> str:
        return "continue"


class NodeBlockStmt(NodeStmtList):
    def __init__(self, type: Type, children: Optional[list[Node]] = None) -> None:
        super().__init__(type, children=children)

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

    def skip_safe_aliases(self) -> Node:
        res = self.src
        while res.type.is_alias():
            res = res.src
        return res

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
        body_str = "{}" if len(field_strs) == 0 else f"{{\n{field_strs}\n}}"
        magic_str = "@magic " if self.sym.magic else ""
        pub_str = "pub " if self.sym.public else ""
        return f"{magic_str}{pub_str}struct {self.sym.render()} {body_str}"

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
        pub_str = "pub " if self.sym.public else ""
        return f"{pub_str}fn {self.sym.render()}({self.params.render()}) -> {self.sym.type.return_type().sym.render()} {self.body.render()}"

class NodeFnCall(Node):
    def __init__(self, callee: Node, original_callee: Node, args: list[Node], type: Type) -> None:
        super().__init__(type)
        self.callee = callee
        self.original_callee = original_callee
        self.args = args

    def render(self) -> str:
        args = ", ".join([x.render() for x in self.args])
        return f"{self.original_callee.render()}({args})"

class NodeIdxOp(Node):
    def __init__(self, lhs: Node, idx: Node, type: Type) -> None:
        super().__init__(type)
        self.lhs = lhs
        self.idx = idx

    def render(self) -> str:
        return f"{self.lhs.render()}[{self.idx.render()}]"

class NodeImport(Node):
    def __init__(self, module_sym: Symbol, orig_path: str, type: Type) -> None:
        super().__init__(type)
        self.module_sym = module_sym
        self.orig_path = orig_path

    def render(self) -> str:
        return f"import {self.orig_path}"

class NodeFromImport(Node):
    def __init__(self, module_sym: Symbol, to_import: list[NodeSymbol], orig_path: str, type: Type, *, wildcard=False) -> None:
        super().__init__(type)
        self.module_sym = module_sym
        self.to_import = to_import
        self.wildcard = wildcard
        self.orig_path = orig_path

    def render(self) -> str:
        if self.wildcard:
            return f"from {self.orig_path} import *"
        else:
            names = ", ".join([x.render() for x in self.to_import])
            return f"from {self.orig_path} import [{names}]"


class ShadowingRule(Enum):
    forbidden = auto() # never allowed to shadow
    allowed = auto() # always allowed to shadow
    shallow = auto() # shallow check allows shadowing inside of a scope


class Symbol:
    def __init__(self, id: str, name: str, shadowing_rule: ShadowingRule, type: Type = None, mutable: bool = False, magic=False, *, source_module: Optional[Module]) -> None:
        self.id = id
        self.name = name
        self.type = type
        self.public = False
        self.mutable = mutable
        self.hidden_mutable = False
        self.definition_node: Node = None
        self.magic = magic
        self._source_module = source_module
        self.shadowing_rule = shadowing_rule

    def comes_from_module(self, module: Module):
        if self._source_module == None:
            return True
        return module == self._source_module

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
    literal_char = auto()

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
    struct = auto()

    module = auto() # Comes from imports: `import x` -> x is a sym of type module
    namespace = auto() # Comes from imports: `import a.b` -> `a` is a namespace and `b` is a module inside of `a`

_int_types = frozenset([
    TypeKind.int8,
    TypeKind.uint8,
    TypeKind.int16,
    TypeKind.uint16,
    TypeKind.int32,
    TypeKind.uint32,
    TypeKind.int64,
    TypeKind.uint64,
])

_ordinal_types = frozenset([
    TypeKind.literal_int,
] + list(_int_types))

_float_types = frozenset([
    TypeKind.float32,
    TypeKind.float64,
])

_literal_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_float,
    TypeKind.literal_bool,
    TypeKind.literal_string,
    TypeKind.literal_char,
])

# TODO: Other solution based on operator signatures
_arith_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_float,
] + list(_int_types) + list(_float_types))

_logical_types = frozenset([
    TypeKind.literal_int,
    TypeKind.literal_bool,
    TypeKind.bool
] + list(_int_types))

_free_infer_types = frozenset([
    TypeKind.magic,
    TypeKind.infer,
])

_callable_types = frozenset([
    TypeKind.function,
    TypeKind.struct,
])


# TODO: Add the other appropriate types
builtin_userspace_types = frozenset(list(_int_types) + list(_float_types) + [TypeKind.bool, TypeKind.char, TypeKind.string, TypeKind.unit])

class Type:
    def __init__(self, kind: TypeKind, sym: Symbol, data: Any = None, *, base: Optional[Type] = None) -> None:
        self.kind = kind
        self.data = data # TODO: arbitrary data for now
        self.sym = sym
        self.base = base
        # TODO: Add a type id later

    def is_alias(self) -> bool:
        return self.kind == TypeKind.alias

    def is_free_infer_type(self) -> bool:
        return self.kind in _free_infer_types

    def is_literal_type(self) -> bool:
        return self.kind in _literal_types

    def is_arith_type(self) -> bool:
        return self.kind in _arith_types

    def is_logical_type(self) -> bool:
        return self.kind in _logical_types

    def is_int_type(self) -> bool:
        return self.kind in _int_types

    def is_float_type(self) -> bool:
        return self.kind in _float_types

    def is_ordinal_type(self) -> bool:
        return self.kind in _ordinal_types

    def is_callable_type(self) -> bool:
        return self.kind in _callable_types

    def skip_safe_aliases(self) -> Type:
        res = self
        while res.is_alias():
            res = res.base
        return res

    def function_arg_types(self) -> list[Type]:
        assert self.kind == TypeKind.function
        return self.data[0]

    def return_type(self) -> Type:
        assert self.kind == TypeKind.function
        return self.data[1]

    def is_indexable(self) -> bool:
        return self.kind in [TypeKind.string, TypeKind.literal_string]

    def element_type(self, graph: ModuleGraph) -> Type:
        assert self.is_indexable()
        match self.kind:
            case TypeKind.string | TypeKind.literal_string:
                return graph.request_module(MAGIC_MODULE_NAME).global_scope.lookup_type("char", shadowing_rule=ShadowingRule.allowed)
            case _:
                raise NotImplementedError()

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
        lhs = self
        rhs = other

        if lhs.is_free_infer_type() or rhs.is_free_infer_type():
            return True

        # TODO: Match variant, like generic inst of generic type
        match lhs.kind:
            case TypeKind.error:
                return False

            # TODO: Not sure what to do about these
            case TypeKind.unit:
                return lhs.kind == rhs.kind

            case TypeKind.function:
                if rhs.kind != TypeKind.function:
                    return False
                lhs_args = lhs.function_arg_types()
                rhs_args = rhs.function_arg_types()
                if len(lhs_args) != len(rhs_args):
                    # TODO: Fix this for optional args
                    return False
                for i in range(0, len(lhs_args)):
                    if not lhs_args[i].types_compatible(rhs_args[i]):
                        return False
                # TODO: Generics?
                return lhs.return_type().types_compatible(rhs.return_type())

            # magic types

            case TypeKind.char | TypeKind.pointer:
                return lhs.kind == rhs.kind
            case TypeKind.bool:
                return lhs.kind == rhs.kind or rhs.kind == TypeKind.literal_bool
            case TypeKind.int8 | TypeKind.uint8 | TypeKind.int16 | TypeKind.uint16 | TypeKind.int32 | TypeKind.uint32 | TypeKind.int64 | TypeKind.uint64:
                return lhs.kind == rhs.kind or rhs.kind == TypeKind.literal_int
            case TypeKind.float32 | TypeKind.float64:
                return lhs.kind == rhs.kind or rhs.kind == TypeKind.literal_float
            case TypeKind.string:
                return lhs.kind == rhs.kind or rhs.kind == TypeKind.literal_string

            case TypeKind.reference:
                if lhs.kind != rhs.kind:
                    return False
                return lhs.data.compare(rhs.data)

            case TypeKind.array:
                if lhs.kind != rhs.kind:
                    return False
                return lhs.data[0].compare(rhs.data[0]) and lhs.data[1].compare(rhs.data[0])

            case TypeKind.static:
                # TODO: static is allowed to be turned into the corresponding non-static version but not vice versa
                raise SerqInternalError("statics aren't ready yet")

            case TypeKind.alias:
                # TODO: Generic aliases. Need to map potentially new generic params around and then reduce the type if required
                raise SerqInternalError("Reached alias comparison for a non-generic alias")

            case TypeKind.distinct:
                # TODO: distinct generics are tough
                raise SerqInternalError("distincts aren't ready yet")


            # user types

            case TypeKind.generic_inst:
                raise SerqInternalError("generic instances aren't ready yet")

            case TypeKind.generic_type:
                raise SerqInternalError("generic types aren't ready yet")

            case TypeKind.struct | TypeKind.concrete_type:
                if id(lhs) == id(rhs): # TODO: It should NOT use python id, should use a concrete id or something. Need a type cache for that
                    return True
                # TODO: If this fails even though the types should be the same the cache got messed up somehow


            # literals

            case TypeKind.literal_bool:
                return rhs.kind in {TypeKind.literal_bool, TypeKind.bool}
            case TypeKind.literal_int:
                return rhs.is_int_type() or rhs.kind == TypeKind.literal_int
            case TypeKind.literal_float:
                return rhs.is_float_type() or rhs.kind == TypeKind.literal_float
            case TypeKind.literal_string:
                return rhs.kind in {TypeKind.literal_string, TypeKind.string}

            case _:
                raise SerqInternalError(f"Unimplemented type comparison: {lhs.kind}")

    def instantiate_literal(self, graph: ModuleGraph) -> Type:
        """
        Turns a literal into a concrete type
        """
        # TODO: Move this out of Type. A type shouldn't instantiate itself for lack of context.
        #       Evident by the lookups below, will break stuff very soon
        assert self.is_literal_type()
        match self.kind:
            case TypeKind.literal_int:
                return graph.request_module(MAGIC_MODULE_NAME).global_scope.lookup_type("int64", shadowing_rule=ShadowingRule.allowed)
            case TypeKind.literal_float:
                return graph.request_module(MAGIC_MODULE_NAME).global_scope.lookup_type("float64", shadowing_rule=ShadowingRule.allowed)
            case TypeKind.literal_bool:
                return graph.request_module(MAGIC_MODULE_NAME).global_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed)
            case TypeKind.literal_string:
                return graph.request_module(MAGIC_MODULE_NAME).global_scope.lookup_type("string", shadowing_rule=ShadowingRule.allowed)
            case _:
                raise SerqInternalError(f"Forgot a literal type: {self.kind}")

    def render(self) -> str:
        if self.kind in builtin_userspace_types or self.is_literal_type():
            return self.kind.name
        elif self.kind == TypeKind.alias:
            return self.sym.render()
        elif self.kind == TypeKind.function:
            args = ", ".join([x.render() for x in self.data[0]])
            return f"fn({args}): {self.data[1].render()}"
        elif self.kind == TypeKind.struct:
            return self.sym.definition_node.render()
        elif self.kind == TypeKind.infer:
            return "<infer>"
        else:
            raise SerqInternalError(f"Render isn't implemented for type kind {self.kind}")


class Scope:
    def __init__(self, graph: ModuleGraph, module: Optional[Module]) -> None:
        self._local_syms: list[Symbol] = []
        self.module_graph = graph # TODO: Get rid of builtin hack
        self.module = module

        # only the oldest sibling should use these
        self._imported_modules: set[Module] = set() # for `import x`
        # this is a dict because they should be ordered in python, so the semantics are valid here
        self._imported_syms: dict[Module, list[Symbol]] = {} # for `from x import [a]`

        # A scope only has access to the immediate syms of their older siblings and their parent which repeats that rule.
        # This prevents out of order access during deferred body transformation.
        # Note: Functions and structs are always added to the oldest sibling to enable mutual dependencies.
        self.parent: Scope = None
        self.older_sibling: Optional[Scope] = None # Can actually be represented as a parent, but this is good enough for now

    def get_oldest_sibling(self) -> Scope:
        if self.older_sibling != None:
            return self.older_sibling.get_oldest_sibling()
        return self

    def iter_module_imports(self, module: Module) -> Iterator[Symbol]:
        oldest = self.get_oldest_sibling()
        for sym in oldest._imported_syms.get(module, []):
            yield sym
        if self.parent != None:
            yield from self.parent.iter_module_imports(module) # TODO: Should "stacked" from imports even be allowed?

    def iter_imported_modules(self) -> Iterator[Module]:
        oldest = self.get_oldest_sibling()
        for module in oldest._imported_modules:
            yield module
        if self.parent != None:
            yield from self.parent.iter_imported_modules()

    def is_imported(self, module: Module, sym: Symbol) -> bool:
        if module in set(self.iter_imported_modules()):
            return True # This is dangerous. It only works because "valid" symbols are guaranteed elsewhere
        for isym in self.iter_module_imports(module):
            if isym == sym:
                return True
        return False

    def _inject_import_namespaces(self, path: str, module_sym: Symbol):
        if path == MAGIC_MODULE_NAME:
            return
        parts = path.split("/")
        if parts[0].startswith("."):
            parts = parts[1:]

        if len(parts) == 1:
            self.get_oldest_sibling().inject(module_sym)
        else:
            if len(parts) == 0:
                raise SerqInternalError("Tried importing an empty path")
            ns = self.put_namespace(parts[0])
            for i in range(1, len(parts) - 1):
                ns = ns.type.data.put_namespace(parts[i])
            ns.type.data.inject(module_sym)

    # TODO: Add a check to prevent conflicts
    def do_basic_import(self, node: NodeImport):
        self._imported_modules.add(node.module_sym.type.data)
        self._inject_import_namespaces(node.orig_path, node.module_sym)

    def do_from_import(self, node: NodeFromImport):
        if node.wildcard:
            syms = list(node.module_sym.type.data.global_scope.iter_syms(shadowing_rule=ShadowingRule.shallow, only_public=True, include_magics=False, include_imports=False))
            self._imported_syms[node.module_sym.type.data] = syms
        else:
            syms: list[Symbol] = []
            for isym in node.to_import:
                if not isinstance(isym, NodeSymbol):
                    raise ValueError("Tried to import a non-symbol")
                syms.append(isym.symbol)
            self._imported_syms[node.module_sym.type.data] = syms
        self._inject_import_namespaces(node.orig_path, node.module_sym)

    def _iter_syms_impl(self, shadowing_rule: ShadowingRule, *, include_magics=True, include_imports=False) -> Iterator[Symbol]:
        # prefer magics
        if self.module_graph.builtin_scope != self and include_magics:
            for sym in self.module_graph.builtin_scope._iter_syms_impl(shadowing_rule=shadowing_rule):
                yield sym

        # local lookup
        for sym in self._local_syms:
            yield sym

        # sibling lookup
        # done in place because we do not want to slip through to parent multiple times
        older = self.older_sibling
        while older != None:
            for sym in older._local_syms:
                yield sym
            if older.older_sibling == None:
                break
            older = older.older_sibling

        # import lookup
        # happens quite late because local syms are always preferred
        if include_imports:
            oldest = older if older != None else self
            for mod, syms in oldest._imported_syms.items():
                for sym in syms:
                    yield sym

        # parent lookup
        if shadowing_rule != ShadowingRule.shallow and self.parent != None:
            yield from self.parent._iter_syms_impl(shadowing_rule=shadowing_rule, include_magics=False, include_imports=include_imports)

    def iter_syms(self, shadowing_rule: ShadowingRule, name: Optional[str] = None, *, include_magics=True, only_public=False, include_imports=False) -> Iterator[Symbol]:
        for sym in self._iter_syms_impl(shadowing_rule=shadowing_rule, include_magics=include_magics, include_imports=include_imports):
            if name != None and sym.name != name:
                continue
            if only_public and not sym.public:
                continue
            yield sym

    def iter_function_defs(self, name: Optional[str] = None, only_public=False, include_imports=True) -> Iterator[Symbol]:
        for sym in self.iter_syms(shadowing_rule=ShadowingRule.allowed, name=name, only_public=only_public, include_imports=include_imports):
            if sym.type.is_callable_type():
                yield sym

    def _lookup_impl(self, name: str, shadowing_rule: ShadowingRule) -> Symbol:
        for sym in self.iter_syms(shadowing_rule=shadowing_rule, name=name, include_imports=True):
            return sym
        return None

    def lookup_typed_sym(self, name: str, expected_type: Type, include_imports=False, only_public=False, include_magics=False) -> NodeSymbol:
        candidates: list[NodeSymbol] = []
        has_fn = False
        for sym in self.iter_syms(shadowing_rule=ShadowingRule.allowed, name=name, include_imports=include_imports, only_public=only_public, include_magics=include_magics):
            if sym.type.is_alias():
                assert isinstance(sym.definition_node, NodeAliasDefinition)
                sym = sym.definition_node.skip_safe_aliases()
            if expected_type == None or expected_type.types_compatible(sym.type) \
                or (expected_type.kind == TypeKind.function and sym.type.kind == TypeKind.struct): # TODO: WORKAROUND! Replace
                candidates.append(NodeSymbol(sym, sym.type))
                if sym.type.kind == TypeKind.function:
                    has_fn = True

        if len(candidates) > 1:
            if has_fn:
                raise ValueError(f"Encountered an ambiguous identifier: {name}")

            block_shadowing = False
            for i in range(len(candidates) - 1, -1, -1):
                rule = candidates[i].symbol.shadowing_rule
                if rule == ShadowingRule.forbidden:
                    block_shadowing = True
                    break
            if block_shadowing:
                # for overload resolution with ambiguous types like literal ints as params
                raise ValueError(f"Encountered an ambiguous identifier: {name}")
            return candidates[0]
        elif len(candidates) == 1:
            return candidates[0]
        else:
            raise ValueError(f"Unable to find identifier {name}")

    def lookup(self, name: str, shadowing_rule: ShadowingRule) -> Optional[Symbol]:
        # Must be unambiguous, can return an unexported symbol. Checked at calltime
        magic = self.module_graph.builtin_scope._lookup_impl(name, shadowing_rule=ShadowingRule.shallow) # TODO: hack
        if magic:
            return magic
        return self._lookup_impl(name, shadowing_rule=shadowing_rule)

    def lookup_type(self, name: str, shadowing_rule: ShadowingRule) -> Optional[Type]:
        # helper for trivial case of sym.type
        sym = self.lookup(name, shadowing_rule=shadowing_rule)
        if sym != None:
            return sym.type

    def inject(self, sym: Symbol):
        self._local_syms.append(sym)


    def put(self, name: str, *, shadowing_rule: ShadowingRule) -> Symbol:
        assert type(name) == str

        if shadowing_rule != ShadowingRule.allowed:
            if self.lookup(name, shadowing_rule=shadowing_rule):
                raise ValueError(f"redefinition of {name}")

        result = Symbol(self.module_graph.sym_id_gen.next(), name=name, shadowing_rule=shadowing_rule, source_module=self.module)
        self.inject(result)
        return result

    # structs are not allowed to shadow
    def put_struct(self, name: str) -> Symbol:
        sym = self.put(name, shadowing_rule=ShadowingRule.forbidden)
        sym.type = Type(kind=TypeKind.struct, sym=sym)
        return sym

    # struct fields are checked shallowly
    def put_struct_field(self, name: str) -> Symbol:
        return self.put(name, shadowing_rule=ShadowingRule.shallow)

    # lets are checked shallowly
    def put_let(self, name: str, mutable: bool) -> Symbol:
        sym = self.put(name, shadowing_rule=ShadowingRule.shallow)
        sym.mutable = mutable
        return sym

    # special type of let, always shadow and fake immutable
    def put_iter_let(self, name: str) -> Symbol:
        sym = self.put(name, shadowing_rule=ShadowingRule.allowed)
        return sym

    # consts are checked shallowly
    def put_const(self, name: str) -> Symbol:
        return self.put(name, shadowing_rule=ShadowingRule.shallow)

    # aliases are checked shallowly
    def put_alias(self, name: str) -> Symbol:
        sym = self.put(name, shadowing_rule=ShadowingRule.shallow)
        sym.type = Type(kind=TypeKind.alias, sym=sym)
        return sym

    # parameters are allowed to shadow
    def put_parameter(self, name: str) -> Symbol:
        return self.put(name, shadowing_rule=ShadowingRule.allowed)

    # functions are allowed to shadow as long as the parameter types are different
    def put_function(self, name: str, typ: Type) -> Symbol:
        assert typ.kind == TypeKind.function
        for fn in self.iter_function_defs(name):
            if fn.type.function_def_args_identical(typ):
                raise ValueError(f"Redefinition of function {name}")
        sym = self.put(name, shadowing_rule=ShadowingRule.allowed)
        sym.type = typ
        return sym

    def put_namespace(self, name: str) -> Symbol:
        existing: Optional[Symbol] = None
        for s in self.iter_syms(shadowing_rule=ShadowingRule.shallow, name=name, include_magics=False):
            if s.type.kind == TypeKind.namespace:
                existing = s
        if existing != None:
            return existing

        sym = self.put(name)
        sym.type = Type(kind=TypeKind.namespace, sym=sym, data=Scope(graph=self.module_graph, module=self.module))
        return sym

    def make_child(self) -> Scope:
        res = Scope(self.module_graph, module=self.module)
        res.parent = self
        return res

    def make_sibling(self) -> Scope:
        res = Scope(self.module_graph, module=self.module)
        res.parent = self.parent
        res.older_sibling = self
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

    def enter_sibling_scope(self):
        self.current_scope = self.current_scope.make_sibling()

    def defer_fn_body(self, sym: Symbol, body: Tree):
        self.module.deferred_fn_bodies.append((sym, body, self.current_scope))


    def get_infer_type(self) -> Type:
        return Type(kind=TypeKind.infer, sym=None)

    def get_unit_type(self) -> Type:
        return self.current_scope.lookup_type("unit", shadowing_rule=ShadowingRule.shallow)


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
            case "for_stmt":
                return self.for_stmt(child, expected_type)
            case "while_stmt":
                return self.while_stmt(child, expected_type)
            case "break_stmt" | "continue_stmt":
                return self.handle_break_or_continue(child, expected_type)
            case "block_stmt":
                return self.handle_block(child, expected_type)
            case "import_stmt":
                return self.handle_import(child)
            case "import_all_from_stmt":
                return self.handle_from_import(child, wildcard=True)
            case "import_from_stmt":
                return self.handle_from_import(child, wildcard=False)
            case _:
                raise SerqInternalError(f"Unimplemented statement type: {child.data}")

    def make_from_import_node(self, import_path: str, names = [], *, wildcard: bool) -> NodeFromImport:
        module = self.graph.request_module(import_path)
        to_import = []
        if not wildcard:
            for ident in names:
                found_syms: list[NodeSymbol] = []
                for sym in module.global_scope.iter_syms(shadowing_rule=ShadowingRule.shallow, name=ident, include_magics=False, only_public=True):
                    found_syms.append(NodeSymbol(sym, sym.type))
                if len(found_syms) == 0:
                    raise ValueError(f"Could not find public symbol `{ident}` in module `{module.name}`")
                to_import.extend(found_syms)
        res = NodeFromImport(
            module_sym=module.sym,
            to_import=to_import,
            orig_path=import_path,
            type=self.get_unit_type(),
            wildcard=wildcard
        )
        self.current_scope.do_from_import(res)
        return res

    def handle_from_import(self, tree: Tree, *, wildcard: bool) -> NodeFromImport:
        import_path = tree.children[0].value
        names = []
        if not wildcard and tree.children[1] != None:
            for ident_node in tree.children[1].children:
                ident = ident_node.children[0].value
                names.append(ident)
        return self.make_from_import_node(import_path=import_path, names=names, wildcard=wildcard)

    def handle_import(self, tree: Tree):
        import_path = tree.children[0].value
        module = self.graph.request_module(import_path)
        res = NodeImport(module.sym, orig_path=import_path, type=self.get_unit_type())
        self.current_scope.do_basic_import(res)
        return res

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

    def range_expression(self, tree: Tree) -> NodeRangeExpr:
        assert tree.data == "range_expression"
        lhs = self.expression(tree.children[0], self.get_infer_type())
        rhs = self.expression(tree.children[1], lhs.type)
        if not lhs.type.is_ordinal_type() or not rhs.type.is_ordinal_type():
            raise ValueError("Only ordinal types can be used in a for loop")
        return NodeRangeExpr(lhs, rhs, self.get_unit_type()) # TODO: Use a range type

    def for_stmt(self, tree: Tree, expected_type: Type) -> NodeStmtList:
        assert tree.data == "for_stmt", tree.data
        assert expected_type.kind == TypeKind.unit
        res = NodeStmtList(self.get_unit_type())

        range_expr = self.range_expression(tree.children[1])
        iter_var = self.current_scope.put_iter_let(tree.children[0].children[0].value)
        iter_var.type = range_expr.start.type
        iter_var.hidden_mutable = True
        iter_var = NodeSymbol(iter_var, iter_var.type)
        cond_node = NodeLessExpression(
            iter_var,
            range_expr.stop,
            self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed)
        )
        res.add(NodeLet(
            iter_var,
            NodeMinusExpression(
                range_expr.start,
                NodeIntLit(1, iter_var.type),
                iter_var.type
            ),
            self.get_unit_type()
        ))

        body = self.handle_block(tree.children[2], self.get_unit_type())
        else_body = NodeBlockStmt(self.get_unit_type(), children=[NodeBreak(self.get_unit_type())])
        body.children = [
            NodeAssignment(
                iter_var,
                NodePlusExpr(
                    iter_var,
                    NodeIntLit(1, iter_var.type),
                    iter_var.type
                ),
                self.get_unit_type()
            ),
            NodeIfStmt(
                cond_expr=cond_node,
                if_body=NodeBlockStmt(body.type, body.children),
                else_body=else_body,
                type=self.get_unit_type()
            )
        ]

        while_loop = NodeWhileStmt(
            NodeBoolLit(True, self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed)),
            body,
            self.get_unit_type()
        )

        res.add(while_loop)
        return res

    def while_stmt(self, tree: Tree, expected_type: Type) -> NodeWhileStmt:
        assert tree.data == "while_stmt", tree.data
        assert expected_type.kind == TypeKind.unit
        # This has to use the outer scope, so a new scope is only opened once this has been checked in full
        while_cond = self.expression(tree.children[0], self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
        assert while_cond.type.kind == TypeKind.bool

        # block_stmt opens a scope
        self.in_loop_counter += 1 # needed to check if break and continue are valid
        body = self.handle_block(tree.children[1], self.get_unit_type())
        self.in_loop_counter -= 1

        return NodeWhileStmt(while_cond, body, self.get_unit_type())

    def if_stmt(self, tree: Tree, expected_type: Type) -> NodeIfStmt:
        assert tree.data == "if_stmt", tree.data
        assert expected_type.kind in [TypeKind.unit, TypeKind.infer] # TODO: if expressions are not unit, need to guarantee valid else branch
        # Same scoping story as in while_stmt
        if_cond = self.expression(tree.children[0], self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
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
            last = self.statement(last_child, expected_type)
            result.add(last)
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

        if expected_type != None:
            if expected_type.is_free_infer_type():
                expected_type = self.current_scope.lookup_type(lookup_name, shadowing_rule=ShadowingRule.allowed)
            else:
                if not expected_type.types_compatible(Type(literal_kind, sym=None)):
                    raise SerqTypeInferError()
            return node_type(value=value, type=expected_type)
        else:
            return node_type(value=value, type=Type(literal_kind, sym=None))

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

    def char(self, tree: Tree, expected_type: Type) -> NodeCharLit:
        assert tree.data == "char", tree.data
        return self.handle_literal(tree, expected_type, "char", TypeKind.literal_char, NodeCharLit, str)


    def binary_expression(self, tree: Tree, expected_type: Type) -> NodeBinaryExpr:
        assert tree.data == "binary_expression", tree.data
        lhs = self.expression(tree.children[0], None)
        # TODO: Can the symbol be captured somehow?
        op = tree.children[1].data
        # a dot expression does not work by the same type rules
        if op != "dot":
            rhs = self.expression(tree.children[2], None)

            lhs_lit = isinstance(lhs, NodeLiteral)
            rhs_lit = isinstance(rhs, NodeLiteral)
            both_lit = lhs_lit and rhs_lit
            if both_lit and type(lhs_lit) != type(rhs_lit):
                raise ValueError("Incompatible literal node types")

            if not lhs.type.types_compatible(rhs.type):
                # TODO: Error reporting
                tl = lhs.type.instantiate_literal(self.graph) if lhs.type.is_literal_type() else lhs.type
                tr = rhs.type.instantiate_literal(self.graph) if rhs.type.is_literal_type() else rhs.type
                raise ValueError(f"Incompatible values in binary expression: `{lhs.render()}`:{tl.render()} {op} `{rhs.render()}`:{tr.render()}")

            # Type coercion. Revisits "broken" nodes and tries to apply the new info on them
            # TODO: Parts of this should probably be pulled into a new function
            expr_type: Type = lhs.type if not lhs.type.is_literal_type() else rhs.type

            # if there is a known type, spread it around
            if lhs.type.is_literal_type() and not rhs.type.is_literal_type():
                lhs = self.expression(tree.children[0], rhs.type)
            elif rhs.type.is_literal_type() and not lhs.type.is_literal_type():
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
                assert expr_type.is_arith_type()
                if both_lit:
                    return type(lhs)(lhs.value + rhs.value, type=lhs.type)
                return NodePlusExpr(lhs, rhs, type=expr_type)
            case "minus":
                assert expr_type.is_arith_type()
                if both_lit:
                    return type(lhs)(lhs.value - rhs.value, type=lhs.type)
                return NodeMinusExpression(lhs, rhs, type=expr_type)
            case "star":
                assert expr_type.is_arith_type()
                if both_lit:
                    return type(lhs)(lhs.value * rhs.value, type=lhs.type)
                return NodeMulExpression(lhs, rhs, type=expr_type)
            case "slash":
                assert expr_type.is_arith_type()
                if both_lit:
                    return type(lhs)(lhs.value / rhs.value, type=lhs.type)
                return NodeDivExpression(lhs, rhs, type=expr_type)

            case "modulus":
                assert expr_type.is_int_type()
                if both_lit:
                    return type(lhs)(lhs.value % rhs.value, type=lhs.type)
                return NodeModExpression(lhs, rhs, type=expr_type)

            case "and":
                assert expr_type.is_logical_type()
                if both_lit:
                    if lhs.type.is_arith_type():
                        return type(lhs)(lhs.value & rhs.value, type=lhs.type)
                    else:
                        return NodeBoolLit(lhs.value and rhs.value, type=expr_type)
                return NodeAndExpression(lhs, rhs, type=expr_type)
            case "or":
                assert expr_type.is_logical_type()
                if both_lit:
                    if lhs.type.is_arith_type():
                        return type(lhs)(lhs.value | rhs.value, type=lhs.type)
                    else:
                        return NodeBoolLit(lhs.value or rhs.value, type=expr_type)
                return NodeOrExpression(lhs, rhs, type=expr_type)

            case "equals":
                if both_lit:
                    return NodeBoolLit(lhs.value == rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
            case "not_equals":
                if both_lit:
                    return NodeBoolLit(lhs.value != rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeNotEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
            case "less":
                # TODO: Ordinal types.
                assert expr_type.is_arith_type()
                if both_lit:
                    return NodeBoolLit(lhs.value < rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeLessExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
            case "lesseq":
                assert expr_type.is_arith_type()
                if both_lit:
                    return NodeBoolLit(lhs.value <= rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeLessEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
            case "greater":
                assert expr_type.is_arith_type()
                if both_lit:
                    return NodeBoolLit(lhs.value > rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeGreaterExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
            case "greatereq":
                assert expr_type.is_arith_type()
                if both_lit:
                    return NodeBoolLit(lhs.value >= rhs.value, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))
                return NodeGreaterEqualsExpression(lhs, rhs, type=self.current_scope.lookup_type("bool", shadowing_rule=ShadowingRule.allowed))

            case "dot":
                lhs = self.expression(tree.children[0], None)
                rhs = tree.children[2].children[0].value

                match lhs.type.kind:
                    case TypeKind.struct:
                        assert isinstance(lhs.type.sym.definition_node, NodeStructDefinition), "Dot operations are currently only ready for structs"
                        matching_field_sym = None
                        # TODO: Do a scope-esque lookup instead. Things like inheritance may silently add more things later
                        for field in lhs.type.sym.definition_node.fields:
                            if not (field.sym.public or field.sym.comes_from_module(self.module)):
                                continue
                            if field.sym.name == rhs:
                                matching_field_sym = field.sym
                                break
                        assert matching_field_sym != None, f"Could not find {rhs} for {lhs.type.sym.render()}"

                        return NodeDotAccess(lhs, NodeSymbol(matching_field_sym, matching_field_sym.type))
                    case TypeKind.module:
                        assert isinstance(lhs.type.data, Module)
                        rhs_sym = lhs.type.data.global_scope.lookup_typed_sym(name=rhs, expected_type=expected_type, only_public=True)
                        return NodeDotAccess(lhs, rhs_sym)
                    case TypeKind.namespace:
                        syms: list[Symbol] = list(lhs.type.data.iter_syms(shadowing_rule=ShadowingRule.allowed, name=rhs, include_magics=False))
                        if len(syms) != 1:
                            raise ValueError(f"Unable to find {rhs} in {lhs.render()}")
                        return NodeDotAccess(lhs, NodeSymbol(syms[0], syms[0].type))
                    case _:
                        raise SerqInternalError(f"Dot operations for type kind `{lhs.type.kind.name}` are not yet allowed")
            case _:
                raise SerqInternalError(f"Unimplemented binary op: {op}")


    def identifier(self, tree: Tree, expected_type: Type) -> NodeSymbol:
        assert tree.data == "identifier", tree.data
        name = tree.children[0].value
        return self.current_scope.lookup_typed_sym(name, expected_type, include_imports=True, include_magics=True)

    def user_type(self, tree: Tree) -> NodeSymbol:
        assert tree.data in ["user_type", "return_user_type"], tree.data
        return self.identifier(tree.children[0], None)

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
            type_sym = self.user_type(type_tree)
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
                assert not val_node.type.is_literal_type()
            # no need to resem the node, it should already be with the current type resolution
        else:
            # infer type from value
            # TODO: Instantiate types, for now only literals
            if val_node.type.kind in builtin_userspace_types or val_node.type.kind in [TypeKind.struct, TypeKind.alias]:
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
        if not isinstance(src, NodeSymbol):
            raise ValueError("Taking an alias of something that isn't a symbol isn't allowed")

        alias_name = tree.children[0].children[0].value
        alias_sym = self.current_scope.put_alias(alias_name)
        alias_sym.type.base = src.type

        res_node = NodeAliasDefinition(alias_sym, src.symbol, self.get_unit_type())
        alias_sym.definition_node = res_node
        return res_node

    def struct_field(self, tree: Tree, expected_type: Type) -> NodeStructField:
        assert tree.data == "struct_field", tree.data
        assert expected_type.kind == TypeKind.unit

        is_public = tree.children[0] != None
        ident = tree.children[1].children[0].value
        sym = self.current_scope.put_struct_field(ident)
        sym.public = is_public

        # TODO: Assert that this is actually a type sym. Current system isn't ready yet
        typ_sym_node = self.user_type(tree.children[2])

        sym.type = typ_sym_node.type

        return NodeStructField(sym, typ_sym_node.symbol.type)

    def struct_definition(self, tree: Tree, expected_type: Type) -> NodeStructDefinition:
        assert tree.data == "struct_definition", tree.data
        assert expected_type.kind == TypeKind.unit

        decorator = tree.children[0].children[0].children[0].value if tree.children[0] != None else ""
        public = tree.children[1] != None

        ident = tree.children[2].children[0].value
        sym = None
        if decorator == "magic":
            sym = self.current_scope.get_oldest_sibling().put_struct(ident)
            sym.type = Type(kind=TypeKind[ident], sym=sym)
            sym.magic = True
        elif decorator == "":
            sym = self.current_scope.get_oldest_sibling().put_struct(ident)
        else:
            raise NotImplementedError("Non-magic struct decorators aren't supported for now")
        sym.public = public

        fields = []
        if tree.children[3] != None:
            self.open_scope()
            for field_node in tree.children[3:]:
                # TODO: Recursion check. Recursion is currently unrestricted and permits infinite recursion
                #   For proper recursion handling we also gotta ensure we don't try to access a type that is currently being processed like will be the case with mutual recursion
                field = self.struct_field(field_node, self.get_unit_type())
                fields.append(field)
            self.close_scope()
        if decorator == "magic" and len(fields) != 0:
            raise ValueError("Tried declaring a magic type with fields")

        def_node = NodeStructDefinition(sym, fields, self.get_unit_type())
        sym.definition_node = def_node
        return def_node

    def handle_deferred_fn_body(self, tree: Tree, sym: Symbol) -> NodeBlockStmt:
        self.current_deferred_ret_type = sym.type.return_type()

        body = self.handle_block(tree, self.get_infer_type())

        # TODO: Revisit this segment once symbol shadowing is in.
        # All of this should be handled by a transformation sym
        if len(body.children) > 0:
            last_body_node = body.children[-1]
            if last_body_node.type.kind == TypeKind.unit and not sym.type.return_type().kind == TypeKind.unit and not isinstance(last_body_node, NodeReturn):
                assert False, f"Returning a unit type for expected type {sym.type.return_type().sym.render()} is not permitted"
            elif isinstance(last_body_node, NodeReturn):
                pass # return is already checked
            elif last_body_node.type.is_literal_type():
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
            sym = self.current_scope.put_parameter(ident)
            type_node = self.user_type(child.children[1])
            sym.type = type_node.symbol.type # TODO: This is not clean at all
            params.append((NodeSymbol(sym, sym.type), type_node))

        return NodeFnParameters(params)

    def fn_definition(self, tree: Tree, expected_type: Type) -> NodeFnDefinition:
        assert tree.data == "fn_definition", tree.data
        assert expected_type.kind == TypeKind.unit

        public = tree.children[0] != None

        ident_node = tree.children[1]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value

        # must open a scope here to isolate the params
        self.open_scope()

        args_node = self.fn_definition_args(tree.children[2], self.get_unit_type())
        ret_type_node = NodeSymbol(self.get_unit_type().sym, self.get_unit_type())
        ret_type = ret_type_node.type
        if tree.children[3] != None:
            ret_type_node = self.user_type(tree.children[3])
            ret_type = ret_type_node.type

        # TODO: Use a type cache to load function with the same type from it for easier matching
        fn_type = Type(
            kind=TypeKind.function,
            sym=None,
            data=([x[1].type for x in args_node.args], ret_type)
        )

        # The sym must be created here to make recursive calls work without polluting the arg scope
        sym = self.current_scope.parent.get_oldest_sibling().put_function(ident, fn_type)
        sym.public = public
        fn_type.sym = sym

        # TODO: Make this work for generics later
        self.defer_fn_body(sym, tree.children[4])

        # Order is important here. We want the sibling to be part of the body's parent to isolate the insides and create a one-way boundary
        self.close_scope()
        self.enter_sibling_scope()

        sym.type = fn_type

        res = NodeFnDefinition(sym, args_node, None, self.get_unit_type())
        sym.definition_node = res
        return res

    def resolve_call(self, callee_node: Node, unresolved_args: list[Tree]) -> Optional[NodeFnCall]:
        def _resolve_call_impl(callee_node: Node, unresolved_args: list[Tree]) -> list[NodeFnCall]:
            match callee_node:
                case NodeSymbol():
                    match callee_node.type.kind:
                        case TypeKind.function:
                            # TODO: This will be wrong with optional args
                            if len(callee_node.type.function_arg_types()) != len(unresolved_args):
                                return []
                            params = []
                            for i in range(0, len(unresolved_args)):
                                formal_type = callee_node.type.function_arg_types()[i]
                                # TODO: Do not eat errors. This makes overload resolution give the wrong error
                                try:
                                    resolved_arg = self.expression(unresolved_args[i], formal_type)
                                    if not resolved_arg.type.types_compatible(formal_type):
                                        return []
                                    params.append(resolved_arg)
                                except SerqTypeInferError as e:
                                    return []
                            return [NodeFnCall(
                                callee=callee_node,
                                original_callee=callee_node, # adjusted by dot case
                                args=params,
                                type=callee_node.type.return_type(),
                            )]
                        case TypeKind.struct:
                            if len(unresolved_args) != 0:
                                raise SerqInternalError("Cannot yet handle calls to types with parameters")
                            return [NodeFnCall(
                                callee=callee_node,
                                original_callee=callee_node, # adjusted by dot case
                                args=[],
                                type=callee_node.type,
                            )]
                case NodeDotAccess():
                    candidates = _resolve_call_impl(callee_node.rhs, unresolved_args)
                    for candidate in candidates:
                        candidate.original_callee = callee_node
                    return candidates
                case _:
                    raise ValueError(f"Calling a {type(callee_node)} is not currently supported")
        # Open a scope here to prevent symbol pollution during param fitting
        # TODO: Not needed once transformation syms exist
        self.open_scope()
        candidates = _resolve_call_impl(callee_node, unresolved_args)
        self.close_scope()
        if len(candidates) > 1:
            raise SerqInternalError("Cannot perform complex function disambiguation yet")
        if len(candidates) == 1:
            return candidates[0]

    def fn_call_expr(self, tree: Tree, expected_type: Type) -> NodeFnCall:
        assert tree.data == "fn_call_expr", tree.data

        unresolved_args: list[Tree] = tree.children[1].children if tree.children[1].children[0] != None else []
        preliminary_arg_types: list[Type] = []
        for unresolved_arg in unresolved_args:
            typ = self.expression(unresolved_arg, None).type
            preliminary_arg_types.append(typ)
        expected_callee_type = Type(TypeKind.function, None, (preliminary_arg_types, self.get_infer_type()))

        callee_node = self.expression(tree.children[0], expected_type=expected_callee_type)
        call = self.resolve_call(callee_node, unresolved_args)
        if call == None:
            raise ValueError(f"No matching overload found for {tree.children[0].children[0].children[0].value}")
        return call

    def idx_op(self, tree: Tree) -> NodeIdxOp:
        assert tree.data == "idx_op", tree.data
        lhs_node = self.expression(tree.children[0], None)
        if not lhs_node.type.is_indexable():
            raise ValueError(f"Tried indexing into a non-indexable expression: {lhs_node.render()}: {lhs_node.type.render()}")
        idx_op = self.expression(tree.children[1], None) # TODO: Any ordinal type
        if not idx_op.type.is_ordinal_type():
            raise ValueError(f"Indexing without an ordinal type is not allowed {idx_op.render()}: {idx_op.type.render()}")
        return NodeIdxOp(lhs_node, idx_op, lhs_node.type.element_type(self.graph))

    def unary_expression(self, tree: Tree, expected_type: Type) -> Node:
        assert tree.data == "unary_expression"
        op = tree.children[0].type
        expr = self.expression(tree.children[1], expected_type)

        if expected_type != None and not expr.type.types_compatible(expected_type):
            raise SerqTypeInferError(f"Value of type {expr.type.render()} is not compatible with expected type {expected_type.render()}")

        match op:
            case "not":
                if not expr.type.is_logical_type():
                    raise SerqTypeInferError(f"Unable to apply `not` to value of type {expr.type.render()}")
                return NodeNotExpression(expr, expr.type)
            case "minus":
                if not expr.type.is_arith_type():
                    raise SerqInternalError(f"Unable to negate value of type {expr.type.render()}")
                return NodeNegExpression(expr, expr.type)
            case _:
                raise NotImplementedError(op)


    def expression(self, tree: Tree, expected_type: Type) -> Node:
        assert tree.data == "expression", tree.data
        assert len(tree.children) == 1
        result = []
        tree: Tree = tree
        for child in tree.children:
            match child.data:
                case "fn_call_expr":
                    result.append(self.fn_call_expr(child, expected_type))
                case "idx_op":
                    result.append(self.idx_op(child))
                case "unary_expression":
                    result.append(self.unary_expression(child, expected_type))
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
                case "char":
                    result.append(self.char(child, expected_type))

                case "identifier":
                    result.append(self.identifier(child, expected_type))
                case _:
                    raise SerqInternalError(f"Unimplemented expression type: {child.data}")
        if len(result) == 1:
            return result[0]
        else:
            raise SerqInternalError(f"Invalid expression length")

    def start(self, tree: Tree) -> NodeStmtList:
        assert tree.data == "start", tree.data
        result = NodeStmtList(self.current_scope.lookup_type("unit", shadowing_rule=ShadowingRule.allowed))
        if self.module.name != MAGIC_MODULE_NAME:
            result.add(self.make_from_import_node(MAGIC_MODULE_NAME, wildcard=True))
        for child in tree.children:
            node = self.statement(child, self.get_unit_type())
            result.add(node)
        if self.current_scope.parent != None:
            raise SerqInternalError("Ended on a scope that isn't top level")
        return result


class Module:
    def __init__(self, name: str, id: int, contents: str, graph: ModuleGraph) -> None:
        self.graph = graph

        self.name = name
        self.global_scope = Scope(graph, module=self)
        self.id = id
        self.hash = hashlib.md5(contents.encode()).digest()
        self.mod_tree = SerqParser(raw_data=contents).parse()
        self.ast: Node = None
        self.deferred_fn_bodies: list[tuple[Symbol, Tree, Scope]] = []
        self.sym: Optional[Symbol] = None

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
        self.modules: dict[pathlib.Path, Module] = {} # TODO: Detect duplicates based on hash, reduce workload
        self._next_id = 0 # TODO: Generate in a smarter way
        self.sym_id_gen = IdGen()

        self.builtin_scope = Scope(self, module=None) # TODO: Remove

        # For now, unit is special because it's overused
        unit_type_sym = self.builtin_scope.put_struct("unit")
        unit_type_sym.magic = True
        unit_type_sym.type = Type(TypeKind.unit, unit_type_sym)

        # TODO: hack
        magic_sym = Symbol("-1", "magic", shadowing_rule=ShadowingRule.forbidden, source_module=None)
        magic_type = Type(TypeKind.magic, magic_sym)
        magic_sym.type = magic_type

        dbg_sym_type = Type(TypeKind.function, None, ([magic_type], unit_type_sym.type))
        dbg_sym = self.builtin_scope.put_function("dbg", dbg_sym_type)
        dbg_sym.magic = True
        dbg_sym_type.sym = dbg_sym

        panic_sym_type = Type(TypeKind.function, None, ([], unit_type_sym.type))
        panic_sym = self.builtin_scope.put_function("panic", panic_sym_type)
        panic_sym.magic = True
        panic_sym_type.sym = panic_sym


    def load(self, path: str | pathlib.Path, file_contents: str) -> Module:
        if isinstance(path, str):
            path = pathlib.Path(path).with_suffix(".serq").absolute()
        name = path.with_suffix("").name

        assert path not in self.modules
        mod = Module(name, self._next_id, file_contents, self)

        # TODO: Proper sym generation
        mod_sym = Symbol(":module:" + str(self._next_id), name, shadowing_rule=ShadowingRule.forbidden, type=None, source_module=None)
        mod_type = Type(TypeKind.module, mod_sym, data=mod)
        mod_sym.type = mod_type
        mod.sym = mod_sym

        self._next_id += 1
        self.modules[path] = mod
        # TODO: Make sure the module isn't already being processed
        ctx = CompCtx(mod, self)
        ast: NodeStmtList = ctx.start(mod.mod_tree)

        # TODO: Check type cohesion
        ctx.handling_deferred_fn_body = True
        old_scope = ctx.current_scope

        while len(mod.deferred_fn_bodies) > 0:
            fn = mod.deferred_fn_bodies.pop()
            ctx.current_scope = fn[2]
            body = ctx.handle_deferred_fn_body(fn[1], fn[0])
            fn[0].definition_node.body = body

        ctx.current_scope = old_scope
        ctx.handling_deferred_fn_body = False

        mod.ast = ast
        return mod

    def request_module(self, name: str) -> Optional[Module]:
        orig_name = name
        is_std_import = name.startswith("std/")
        if is_std_import:
            name = "./serqlib/std/" + name[4:]
        path = pathlib.Path(name).with_suffix(".serq").absolute()

        if path in self.modules:
            return self.modules[path]
        elif path.is_file():
            return self.load(name, path.read_text())
        else:
            raise ValueError(f"Unable to find import: {orig_name}")
