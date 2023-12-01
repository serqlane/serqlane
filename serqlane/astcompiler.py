from __future__ import annotations
from typing import Any, Optional
from enum import Enum, auto

import hashlib

import lark.visitors
from lark import Tree, Token

from serqlane.parser import SerqParser


class Node:
    def __init__(self, type: Type = None) -> None:
        self.type = type

    def render(self, indent=0) -> str:
        raise NotImplementedError()

class NodeStmtList(Node):
    def __init__(self) -> None:
        super().__init__()
        self.children: list[Node] = []

    def add(self, node: Node):
        self.children.append(node)

    def render(self, indent=0) -> str:
        result = ""
        for child in self.children:
            result += child.render() + "\n"
        return result

class NodeFnDefinition(Node):
    def __init__(self, public: bool, sym: Symbol, params: Node, body: Node) -> None:
        super().__init__()
        self.public =  public
        self.sym = sym
        self.params = params
        self.body = body

class NodeLiteral[T](Node):
    def __init__(self, value: T, type: Type) -> None:
        super().__init__(type) # type either gets converted from literal to actual, gets turned into an error or gets inferred from lhs
        self.value = value

    def render(self) -> str:
        return str(self.value)

class NodeIntLit(NodeLiteral[int]): ...
class NodeFloatLit(NodeLiteral[float]): ...
class NodeBoolLit(NodeLiteral[bool]): ...
class NodeStringLit(NodeLiteral[str]):
    def render(self) -> str:
        return f"\"{self.value}\""

class NodeLet(Node):
    def __init__(self, sym: Symbol, type_sym: Symbol, expr: Node):
        super().__init__()
        self.sym = sym
        self.type_sym = type_sym # TODO: store as a type hint node instead of sym
        self.expr = expr

    def render(self, indent=0) -> str:
        is_mut = self.sym.mutable
        return f"{" " * indent}let {"mut " if is_mut else ""}{self.sym.name}{": " + self.type_sym.render() if self.type_sym != None else ""} = {self.expr.render()};"

class Symbol:
    def __init__(self, name: str, type: Type = None, exported: bool = False, mutable: bool = False) -> None:
        # TODO: Should store the source node, symbol kinds, sym id
        self.name = name
        self.type = type
        self.exported = exported
        self.mutable = mutable

    def render(self) -> str:
        # TODO: Use type info to render generics and others
        return self.name


class TypeKind(Enum):
    error = auto() # bad type
    empty = auto() # type that hasn't yet been filled. might be because it needs to be inferred
    
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

# TODO: Add the other appropriate types
builtin_userspace_types = frozenset(list(int_types) + list(float_types) + [TypeKind.bool, TypeKind.char, TypeKind.string])

class Type:
    def __init__(self, kind: TypeKind, sym: Symbol, data: Any = None) -> None:
        self.kind = kind
        self.data = data # TODO: arbitrary data for now
        self.sym = sym
        # TODO: Add a type id later

    def can_convert_into(self, other: Type) -> bool:
        """
        other is always the target
        """
        # TODO: Match variant, like generic inst of generic type
        match self.kind:
            case TypeKind.error:
                return False
            
            case TypeKind.empty:
                # types that are not allowed to be inferred
                if other.kind in {TypeKind.error, TypeKind.empty}:
                    return False
                else:
                    return True
            
            # TODO: Not sure what to do about these
            case TypeKind.unit:
                raise ValueError("units aren't ready yet")
            

            # magic types

            # TODO: Can you match sets?
            case TypeKind.bool | TypeKind.char| \
                TypeKind.int8 | TypeKind.uint8 | TypeKind.int16 | TypeKind.uint16 | TypeKind.int32 | TypeKind.uint32 | TypeKind.int64 | TypeKind.uint64 | \
                TypeKind.float32 | TypeKind.float64 | TypeKind.pointer | TypeKind.string:
                return self.kind == other.kind
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
                return other.kind == TypeKind.bool
            case TypeKind.literal_int:
                return other.kind in int_types
            case TypeKind.literal_float:
                return other.kind in float_types
            case TypeKind.literal_string:
                return other.kind == TypeKind.string

            case _:
                raise ValueError(f"Unimplemented type comparison: {self.kind}")

    def render(self) -> str:
        # TODO: match on sets?
        if self.kind in builtin_userspace_types or self.kind in literal_types:
            return self.kind.name
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
        return self._lookup_impl(name)

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
        
        result = Symbol(name)
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

    def make_sibling() -> Scope:
        pass

    def make_child() -> Scope:
        pass

    def merge(other: Scope):
        pass


class CompCtx(lark.visitors.Interpreter):
    def __init__(self, module: Module, graph: ModuleGraph) -> None:
        self.module = module
        self.graph = graph
        self.current_scope = self.module.global_scope

    def integer(self, tree: Tree):
        val = tree.children[0].value
        # untyped at this stage
        return NodeIntLit(value=int(val), type=Type(TypeKind.literal_int, sym=None))
    
    def decimal(self, tree: Tree):
        val = f"{tree.children[0].children[0].value}.{tree.children[1].children[0].value}"
        return NodeFloatLit(value=float(val), type=Type(TypeKind.literal_float, sym=None))
    
    def bool(self, tree: Tree):
        val = tree.children[0].value
        return NodeBoolLit(value = val == "true", type=Type(TypeKind.literal_bool, sym=None))
    
    def string(self, tree: Tree):
        val = tree.children[0].value
        return NodeStringLit(value=val, type=Type(TypeKind.literal_string, sym=None))

    def identifier(self, tree: Tree):
        val = tree.children[0].value
        sym = self.current_scope.lookup(val)
        if sym:
            return sym
        # TODO: Error reporting
        raise ValueError(f"Bad identifier: {val}")

    def user_type(self, tree: Tree):
        return self.visit(tree.children[0])

    def let_stmt(self, tree: Tree):
        mut_node = tree.children[0]

        # currently the only modifier
        is_mut = isinstance(mut_node, Token)
        if isinstance(mut_node, Token):
            assert mut_node.type == "MUT"

        f = int(is_mut)
        ident_node = tree.children[f]
        assert ident_node.data == "identifier"
        ident = ident_node.children[0].value
        sym = self.current_scope.put_let(ident, mutable=is_mut)

        f += 1

        type_sym = None
        if tree.children[f].data == "user_type":
            # we have a user provided type node
            type_tree = tree.children[f]
            type_sym = self.visit(type_tree) # TODO: Return an empty node with empty type
            assert isinstance(type_sym, Symbol)
            f += 1
        
        val_node = self.visit(tree.children[f])

        if type_sym != None:
            # check types for compatibility
            if not val_node.type.can_convert_into(type_sym.type):
                # TODO: Error reporting
                raise ValueError(f"Variable type {type_sym.name} is not compatible with value of type {val_node.type.render()}")
            val_node.type = type_sym.type # coerce the node type
        else:
            # infer type from value
            # TODO: Instantiate types, for now only literals
            #type_sym = val_node.type
            pass

        f += 1
        assert len(tree.children) == f

        return NodeLet(
            sym=sym,
            type_sym=type_sym,
            expr=val_node
        )


    def expression(self, tree: Tree):
        # TODO: Handle longer expressions
        assert len(tree.children) == 1
        result = []
        tree: Tree = tree
        for child in tree.children:
            result.append(self.visit(child))
        if len(result) == 1:
            return result[0]
        else:
            assert False

    def start(self, tree: Tree):
        result = NodeStmtList()
        for child in tree.children:
            node = self.visit(child)
            assert len(node) == 2 and node[1] == []
            result.add(node[0])
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
        # mapping of (name, dict[params]) -> Type
        self.generic_cache: dict[(str, dict[Symbol, Type]), Type] = {}

    def lookup_toplevel(self, name: str) -> Optional[Symbol]:
        return self.global_scope.lookup(name, shallow=True)


class ModuleGraph:
    def __init__(self) -> None:
        self.modules: dict[str, Module] = {} # TODO: Detect duplicates based on hash, reduce workload
        self._next_id = 0 # TODO: Generate in a smarter way

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
        ast: NodeStmtList = CompCtx(mod, self).visit(mod.lark_tree)
        print(ast.render()) # TODO: Only for debugging

        return mod
