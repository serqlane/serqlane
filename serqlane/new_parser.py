from __future__ import annotations

import pathlib

from typing import Optional

from serqlane.common import SerqInternalError
from serqlane.tokenizer import Tokenizer, SqTokenKind, SqToken


class ParserError(Exception): ...


class Token:
    def __init__(self, data: str, value: str):
        self.data = data
        self.value = value

class Tree:
    def __init__(self, data: str, *, children: Optional[list[Tree | Token]] = None):
        self.data = data
        self.children: list[Tree | Token] = children if children != None else []

    def add(self, x: Tree | Token):
        self.children.append(x)


class SerqParser:
    def __init__(self, raw_data: str, *, filepath: Optional[pathlib.Path] = None) -> None:
        self.filepath = filepath
        self.tokenizer = Tokenizer(raw_data, filepath=filepath)
        self._token_queue: list[SqToken] = [] # stores the current token and however many were peeked
        self._token_iter = self.tokenizer.process_iter()
    
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

    def _handle_identifier(self, tok: SqToken) -> Tree:
        assert tok.kind == SqTokenKind.IDENTIFIER
        return Tree("identifier", children=[Token("identifier", tok.literal)])

    def _handle_expression(self) -> Tree:
        result = Tree("expression")
        cursor = self.advance()

        match cursor.kind:
            case SqTokenKind.INTEGER:
                result.add(Tree("integer", children=[Token("integer", cursor.literal)]))
            case _:
                raise NotImplementedError(cursor.kind)

        return result

    def _handle_let(self) -> Tree:
        result = Tree("let_stmt")

        ident = self.expect([SqTokenKind.MUT, SqTokenKind.IDENTIFIER])
        if ident.kind == SqTokenKind.MUT:
            result.add(Token("MUT", ident.literal))
            ident = self.expect([SqTokenKind.IDENTIFIER])
        result.add(self._handle_identifier(ident))
        self.expect([SqTokenKind.EQ])
        result.add(self._handle_expression())

        return result
    
    def _handle_decorator(self) -> Tree:
        result = Tree("decorator")
        result.add(self._handle_identifier(self.expect([SqTokenKind.IDENTIFIER])))
        return result

    def _handle_decoratable(self, *, decorator: Optional[Tree]) -> Tree:
        result = Tree("<placeholder>")

        result.add(decorator)

        cursor = self.expect([SqTokenKind.PUB, SqTokenKind.STRUCT, SqTokenKind.FN])
        if cursor.kind == SqTokenKind.PUB:
            result.add(Token("PUB", "pub"))
            cursor = self.expect([SqTokenKind.STRUCT, SqTokenKind.FN])
        if cursor.kind == SqTokenKind.STRUCT:
            result.data = "struct_definition"
        elif cursor.kind == SqTokenKind.FN:
            result.data = "fn_definition"
        else: raise NotImplementedError(cursor.kind)
        cursor = self.expect([SqTokenKind.IDENTIFIER])
        result.add(self._handle_identifier(cursor))

        self.expect([SqTokenKind.OPEN_CURLY])
        result.add(None)
        self.expect([SqTokenKind.CLOSE_CURLY])
        
        return result

    def _handle_statement(self, start_token: SqToken) -> Tree:
        result = Tree("statement")
        
        match start_token.kind:
            case SqTokenKind.LET:
                result.add(self._handle_let())
            case SqTokenKind.AT:
                decorator = self._handle_decorator()
                inner = self._handle_decoratable(decorator=decorator)
                result.add(inner)
            case _:
                raise NotImplementedError(start_token.kind)

        return result

    def parse(self) -> Tree:
        result = Tree("start")
        tok = self.advance()
        while tok.kind != SqTokenKind.EOF:
            result.add(self._handle_statement(tok))
            tok = self.advance()
        return result


if __name__ == "__main__":
    t = SerqParser("""
@decorator
pub struct int8 {}
    """).parse()
    print([x.data for x in t.children])
