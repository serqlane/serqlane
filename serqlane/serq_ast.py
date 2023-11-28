from __future__ import annotations

from dataclasses import dataclass

from enum import Enum


class AstNode: ...


class Expression(AstNode): ...


@dataclass
class Identification(AstNode):
    name: str


@dataclass
class Conditional(Expression):
    condition: Expression
    then: Expression


@dataclass
class Block(Expression):
    lines = list[Expression]


@dataclass
class ConditionalBlock(Expression):
    branches = list[Conditional]
    else_branch = Expression | None


class Operator(Enum):
    add = "+"
    sub = "-"
    divide = "/"
    multiply = "*"


@dataclass
class Operation(Expression):
    left_hand: Expression
    operator: Operator
    right_hand: Expression


class LiteralType(Enum):
    number = 1
    bool = 2
    string = 3


@dataclass
class Literal(Expression):
    value: str
    type: LiteralType


@dataclass
class Argument(AstNode):
    name: Identification
    type: Identification


@dataclass
class Function(AstNode):
    name: Identification
    args: list[Argument]
    return_type: Identification
    body: Block
