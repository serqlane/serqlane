from __future__ import annotations

from typing import Any, Optional
from enum import Enum, auto

import hashlib

import lark.visitors

from serqlane.parser import SerqParser


class Node:
    def __init__(self, type: Type | None = None) -> None:
        self.type = type

    def render(self, indent=0) -> str:
        raise NotImplementedError()

class NodeStmtList(Node):
    def __init__(self) -> None:
        super().__init__()
        self.children: list[Node] = []

    def add(self, node: Node):
        self.children.append(node)

class NodeFnDefinition(Node):
    def __init__(self, public: bool, sym: Symbol, params: Node, body: Node) -> None:
        super().__init__()
        self.public =  public
        self.sym = sym
        self.params = params
        self.body = body

class NodeLiteral(Node):
    def __init__(self, value, type: Type | None = None) -> None:
        super().__init__(type)
        self.value = value

    def render(self, indent=0) -> str:
        return str(self.value)

class NodeLet(Node):
    def __init__(self, sym: Symbol, expr: Node):
        super().__init__()
        self.sym = sym
        self.expr = expr

    def render(self, indent=0) -> str:
        is_mut = self.sym.mutable
        return f"{" " * indent}let {"mut " if is_mut else ""}{self.sym.name} = {self.expr.render()}"


class Symbol:
    def __init__(self, name: str, type: Type | None = None, exported: bool = False, mutable: bool = False) -> None:
        # TODO: Should store the source node, symbol kinds, sym id
        self.name = name
        self.type = type
        self.exported = exported
        self.mutable = mutable


class TypeKind(Enum):
    error = auto() # bad type

    unit = auto() # zero sized type
    never = auto() # guaranteed to terminate the current scope

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
    slice = auto() # TODO: &slice(len)[T]

    alias = auto() # Alias[T]
    distinct = auto() # distinct[T]
    concrete_type = auto() # non-generic concrete Type or fn()
    generic_inst = auto() # fully instantiated generic type Type[int] or fn[int](): generic_inst[concerete_type, generic_type[params]]
    generic_type = auto() # Type[T] or fn[T](): generic_type[params]
    type = auto() # TODO: magic that holds a type itself, not yet in grammar

    bool_lit = auto()
    int_lit = auto()
    float_lit = auto()
    str_lit = auto()


class Type:
    def __init__(self, kind: TypeKind, data: Any = None) -> None:
        self.kind = kind
        self.data = data # TODO: arbitrary data for now
        # TODO: Add a type id later

    def compare(self, other: Type) -> bool:  # type: ignore (type checker doesn't account for _ case of match)
        """
        other is always the target
        """
        # TODO: Match variant, like generic inst of generic type
        match self.kind:
            case TypeKind.error:
                return False
            
            # TODO: Not sure what to do about these
            case TypeKind.unit:
                raise ValueError("units aren't ready yet")
            case TypeKind.never:
                raise ValueError("nevers aren't ready yet")
            

            # magic types

            case TypeKind.bool | TypeKind.char| \
                TypeKind.int8 | TypeKind.uint8 | TypeKind.uint16 | TypeKind.int32 | TypeKind.uint32 | TypeKind.int64 | TypeKind.uint64 | \
                TypeKind.float32 | TypeKind.float64 | TypeKind.pointer | TypeKind.string:
                # TODO: Handle literals
                return self.kind == other.kind
            
            case TypeKind.reference, TypeKind.slice:
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

            case TypeKind.bool_lit:
                return other.kind == TypeKind.bool
            case TypeKind.int_lit:
                return other.kind in { TypeKind.int8, TypeKind.uint8, TypeKind.int16, TypeKind.uint16, TypeKind.int32, TypeKind.uint32, TypeKind.int64, TypeKind.uint64 }
            case TypeKind.float_lit:
                return other.kind in { TypeKind.float32, TypeKind.float64 }
            case TypeKind.str_lit:
                return other.kind == TypeKind.string

            case _:
                raise ValueError(f"Unimplemented type comparison: {self.kind}")


class Scope:
    def __init__(self) -> None:
        self._local_syms: list[Symbol] = []
        self.parent: Scope | None = None

    def lookup(self, name: str, shallow=False) -> Symbol | None:
        # Must be unambiguous, can return an unexported symbol. Checked at calltime
        for symbol in self._local_syms:
            if symbol.name == name:
                return symbol
        if shallow or self.parent == None:
            # TODO: Report error in module
            return None
        return self.lookup(name)

    def put(self, name: str) -> Symbol:
        assert type(name) == str
        if self.lookup(name): raise ValueError(f"redefinition of {name}")
        
        result = Symbol(name)
        self._local_syms.append(result)
        return result

    def make_sibling(self) -> Scope:
        ...

    def make_child(self) -> Scope:
        ...

    def merge(self, other: Scope):
        ...


class CompCtx(lark.visitors.Interpreter):
    def __init__(self, module: Module, graph: ModuleGraph) -> None:
        self.module = module
        self.graph = graph
        self.current_scope = self.module.global_scope


class Module:
    def __init__(self, name: str, id: int, contents: str, graph: ModuleGraph) -> None:
        self.graph = graph
        # TODO: Naming conflicts
        self.imported_modules: list[Module] = []

        self.name = name
        self.global_scope = Scope()
        self.id = id
        self.hash = hashlib.md5(contents.encode()).digest()
        self.lark_tree = SerqParser().parse(contents, display=False)
        # mapping of (name, dict[params]) -> Type
        self.generic_cache: dict[tuple[str, dict[Symbol, Type]], Type] = {}

    def lookup_toplevel(self, name: str) -> Optional[Symbol]:
        return self.global_scope.lookup(name, shallow=True)


class ModuleGraph:
    def __init__(self) -> None:
        self.modules: dict[str, Module] = {} # TODO: Detect duplicates based on hash, reduce workload
        self._next_id = 0 # TODO: Generate in a smarter way

    def load(self, name: str, file_contents: str) -> Module:
        assert name not in self.modules
        mod = Module(name, self._next_id, file_contents, self)
        self._next_id += 1
        self.modules[name] = mod
        
        # TODO: Make sure the module isn't already being processed
        ast = CompCtx(mod, self).visit(mod.lark_tree)
        print(ast[0][0].render()) # TODO: Only for debugging

        return mod
