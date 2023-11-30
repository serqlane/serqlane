from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Self

from lark import Token, Tree

from serqlane.parser import SerqParser


@dataclass
class Wrapper:
    value: Any
    mutable: bool


@dataclass
class FunctionWrapper(Wrapper):
    scope: Scope


class Scope:
    def __init__(
        self, parent: Scope | None = None, *, values: dict[str, Wrapper] | None = None
    ) -> None:
        self.parent = parent
        self.values: dict[str, Wrapper] = values or {}

    def lookup(self, name: str) -> Wrapper:
        try:
            return self.values[name]
        except KeyError:
            parent = self.parent
            while parent is not None:
                try:
                    return parent.lookup(name)
                except ValueError:
                    parent = parent.parent

            raise ValueError(f"{name} not found in scope")

    def add_function(self, name: str, tree: Tree[Token]):
        if self.values.get(name) is not None:
            raise ValueError(f"{name} is already set")

        self.values[name] = FunctionWrapper(tree, mutable=False, scope=self.copy())

    def add(self, name: str, value: Any, mutable: bool = False):
        if self.values.get(name) is not None:
            raise ValueError(f"{name} is already set")

        self.values[name] = Wrapper(value, mutable)

    def set(self, name: str, value: Any):
        """Modify mutable value

        Args:
            name (str): name of the value to modify
            value (Any): value to set as
        """
        try:
            wrapper = self.values[name]
        except KeyError:
            raise KeyError(f"{name} is not in scope")
        else:
            if not wrapper.mutable:
                raise ValueError(f"{name} is not mutable")

            wrapper.value = value

    def copy(self) -> Self:
        return type(self)(parent=self.parent, values=self.values.copy())


class SerqVM:
    def __init__(self) -> None:
        self.current_scope: Scope = Scope()
        self.stack: list[Tree[Token]] = []

    # def run(self, tree: Tree[Token]):
    #     for child in tree.children:
    #         if isinstance(child, Tree):
    #             base: Token = child.data # type: ignore
    #         else:
    #             base = child

    #         print(f"{child.data=}")
    #         print(f"{base.type=}")

    #         match base.value:
    #             case "define":
    #                 self.do_define(child) # type: ignore

    # def do_define(self, tree: Tree[Token]):
    #     children = tree.children
    #     if isinstance(children[0], Token):
    #         mutable = children[0].type == "MUT"

    #     print(f"{mutable=} {tree.children=}")

    # def do_assignment(self, tree: Tree[Token]) -> Any:
    #     name = tree.children[0].value # type: ignore

    # def evaluate_expression(self):
    #     ...


if __name__ == "__main__":
    test = "let x = 1;"
    parser = SerqParser()
    tree = parser.parse(test)
    vm = SerqVM()
    #vm.run(tree)
