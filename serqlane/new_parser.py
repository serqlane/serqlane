from __future__ import annotations

import pathlib

from typing import Optional, Callable

from serqlane.common import SerqInternalError
from serqlane.tokenizer import Tokenizer, SqTokenKind, SqToken


class ParserError(Exception): ...


class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value

class Tree:
    def __init__(self, data: str, *, children: Optional[list[Tree | Token | None]] = None):
        self.data = data
        self.children: list[Tree | Token | None] = children if children != None else []

    def add(self, x: Tree | Token | None):
        self.children.append(x)


class SerqParser:
    def __init__(self, raw_data: str, *, filepath: Optional[pathlib.Path] = None) -> None:
        self.filepath = filepath
        self.tokenizer = Tokenizer(raw_data, filepath=filepath)
        self._token_queue: list[SqToken] = [] # stores the current token and however many were peeked
        self._token_iter = self.tokenizer.process_iter()
        
        self._cur_decorator: Optional[Tree] = None
        self._cur_pub: Optional[Tree] = None

    def peek(self, offset=1) -> SqToken:
        diff = offset + 1 - len(self._token_queue)
        while diff > 0:
            self._token_queue.append(self._token_iter.__next__())
            diff -= 1
        return self._token_queue[offset]

    def advance(self) -> SqToken:
        tok = self.peek(0)
        self._token_queue = self._token_queue[1:]
        return tok

    def expect(self, kinds: list[SqTokenKind]) -> SqToken:
        result = self.advance()
        if result.kind not in kinds:
            raise ParserError(f"Expected token kinds {kinds} but got {result.kind}")
        return result

    def skip(self, kinds: list[SqTokenKind]):
        t = self.peek(0)
        while t.kind in kinds:
            self.advance()
            t = self.peek(0)

    def _make_identifier(self, ident: SqToken) -> Tree:
        return Tree("identifier", children=[Token("identifier", ident.literal)])

    def _eat_identifier(self) -> Tree:
        ident = self.expect([SqTokenKind.IDENTIFIER])
        return self._make_identifier(ident)

    def _eat_operator(self) -> Token:
        op = self.expect([
            SqTokenKind.PLUS,
            SqTokenKind.MINUS,
        ])
        match op.kind:
            case SqTokenKind.PLUS | SqTokenKind.MINUS:
                return Token(op.kind.name.lower(), op.kind.value)
            case _:
                raise NotImplementedError(op.kind)

    def _descend_atom_expr(self) -> Tree:
        x = self.expect([
            SqTokenKind.INTEGER,
            SqTokenKind.DECIMAL,
            SqTokenKind.STRING,
            SqTokenKind.TRUE,
            SqTokenKind.FALSE,
            SqTokenKind.IDENTIFIER,
        ])
        match x.kind:
            case SqTokenKind.INTEGER | SqTokenKind.DECIMAL | SqTokenKind.STRING:
                name = x.kind.name.lower()
                return Tree(name, children=[Token(name, x.literal)])
            case SqTokenKind.TRUE | SqTokenKind.FALSE:
                return Tree("bool", children=[Token(x.literal, x.literal)])
            case SqTokenKind.IDENTIFIER:
                return self._make_identifier(x)
            case _:
                raise NotImplementedError(x.kind)


    def _descend_plus_expr(self) -> Tree:
        lhs = self._descend_atom_expr()
        op = self.peek(0)
        if op.kind in [SqTokenKind.PLUS, SqTokenKind.MINUS]:
            op = self._eat_operator()
            rhs = self._descend_atom_expr()
            return Tree("binary_expr", children=[lhs, op, rhs])
        else:
            return lhs

    def _descend_binary_expr(self) -> Tree:
        return self._descend_plus_expr()

    def _eat_expression(self) -> Tree:
        return Tree("expression", children=[self._descend_binary_expr()])


    def _eat_decorator(self) -> Tree:
        self.expect([SqTokenKind.AT])
        if self._cur_decorator != None:
            raise ParserError("Using multiple decorators isn't currently allowed")
        ident = self._eat_identifier()
        self._cur_decorator = Tree("decorator", children=[ident])
        return self._cur_decorator

    def _eat_mut(self) -> Token:
        self.expect([SqTokenKind.MUT])
        return Token("MUT", "mut")

    def _eat_let(self) -> Tree:
        self.expect([SqTokenKind.LET])
        res = Tree("let_stmt")
        if self.peek(0).kind == SqTokenKind.MUT:
            res.add(self._eat_mut())
        res.add(self._eat_identifier())
        self.expect([SqTokenKind.EQ])
        res.add(self._eat_expression())
        return res
    
    def _eat_struct(self) -> Tree:
        self.expect([SqTokenKind.STRUCT])
        result = Tree("struct_definition", children=[self._cur_decorator, self._cur_pub])
        self._cur_pub = None
        self._cur_decorator = None

        result.add(self._eat_identifier())
        self.expect([SqTokenKind.OPEN_CURLY])
        result.add(None) # TODO: Implement fields
        self.expect([SqTokenKind.CLOSE_CURLY])

        return result
    
    def _eat_statement(self) -> Optional[Tree]:
        tok = self.peek(0)
        result = Tree("statement")
        match tok.kind:
            case SqTokenKind.AT:
                self._eat_decorator()
                return None
            case SqTokenKind.PUB:
                self.advance()
                if self._cur_pub != None:
                    raise ParserError("Encountered multiple pub tokens without a consumer")
                self._cur_pub = Tree("PUB", children=[Token("PUB", "pub")])
                return None
            case SqTokenKind.STRUCT:
                result.add(self._eat_struct())
            case SqTokenKind.LET:
                if self._cur_decorator != None:
                    raise ParserError("Decorators aren't allowed for let statements")
                result.add(self._eat_let())
            case _:
                expr = self._eat_expression()

                if self.peek(0).kind == SqTokenKind.EQ:
                    self.advance()
                    rhs = self._eat_expression()
                    result.add(Tree("assignment", children=[expr, rhs]))
                else:
                    raise NotImplementedError(tok.kind)
        if len(result.children) == 0:
            raise SerqInternalError()
        return result

    def parse(self) -> Tree:
        result = Tree("start")
        tok = self.peek(0)
        while tok.kind != SqTokenKind.EOF:
            stmt = self._eat_statement()
            if stmt != None:
                result.add(stmt)
            tok = self.peek(0)
        return result


if __name__ == "__main__":
    t = SerqParser("""
@decorator
pub struct int8 {}
    """).parse()
    print([x for x in t.children])
