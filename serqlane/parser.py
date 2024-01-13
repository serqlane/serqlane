from __future__ import annotations

import pathlib

from typing import Optional

from serqlane.common import SerqInternalError
from serqlane.tokenizer import Tokenizer, SqTokenKind, SqToken


class ParserError(Exception): ...


class Token:
    def __init__(self, type: str, value: str):
        self.type = type
        self.data = type
        self.value = value

    def __repr__(self) -> str:
        return f"{self.type}"

class Tree:
    def __init__(self, data: str, *, children: Optional[list[Tree | Token | None]] = None):
        self.data = data
        self.children: list[Tree | Token | None] = children if children != None else []

    def add(self, x: Tree | Token | None):
        self.children.append(x)

    def __repr__(self) -> str:
        children = ", ".join([repr(x) for x in self.children])
        return f"{self.data}: [{children}]"


class SerqParser:
    def __init__(self, raw_data: str, *, filepath: Optional[pathlib.Path] = None) -> None:
        self.filepath = filepath
        self.tokenizer = Tokenizer(raw_data, filepath=filepath)
        self._token_queue: list[SqToken] = [] # stores the current token and however many were peeked
        self._token_iter = self.tokenizer.process_iter()
        
        self._cur_decorator: Optional[Tree] = None
        self._cur_pub: Optional[Tree] = None
        
        self._skip_newlines = True
        self._stored_newlines = 0

    def peek(self, offset: int = 1) -> SqToken:
        diff = offset + 1 - len(self._token_queue) + self._stored_newlines
        while diff > 0:
            tok = self._token_iter.__next__()
            self._token_queue.append(tok)
            if tok.kind == SqTokenKind.NEWLINE:
                self._stored_newlines += 1
            else:
                diff -= 1
        if self._skip_newlines:
            i = 0
            ctr = -1
            tok: Optional[SqToken] = None
            while ctr != offset:
                tok = self._token_queue[i]
                i += 1
                if tok.kind != SqTokenKind.NEWLINE or not self._skip_newlines:
                    ctr += 1
            if tok == None:
                raise SerqInternalError()
            return tok
        else:
            return self._token_queue[offset]

    def advance(self) -> SqToken:
        should_skip = self._skip_newlines
        self._skip_newlines = False
        tok = self.peek(0)
        if should_skip:
            while tok.kind == SqTokenKind.NEWLINE:
                self._token_queue = self._token_queue[1:]
                self._stored_newlines -= 1
                tok = self.peek(0)
        else:
            if tok.kind == SqTokenKind.NEWLINE:
                self._stored_newlines -= 1
        self._token_queue = self._token_queue[1:]
        self._skip_newlines = should_skip
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

    def _wrap_expr(self, expr: Tree) -> Tree:
        if expr.data == "expression":
            return expr
        else:
            return Tree("expression", children=[expr])

    def _eat_identifier(self) -> Tree:
        ident = self.expect([SqTokenKind.IDENTIFIER])
        return self._make_identifier(ident)

    def _eat_user_type(self) -> Tree:
        self.expect([SqTokenKind.COLON])
        return Tree("user_type", children=[self._eat_identifier()])

    def _eat_return_user_type(self) -> Tree:
        self.expect([SqTokenKind.ARROW])
        return Tree("return_user_type", children=[self._eat_identifier()])

    def _eat_function_args(self) -> Tree:
        result = Tree("fn_call_args")
        self.expect([SqTokenKind.OPEN_PAREN])

        while self.peek(0).kind != SqTokenKind.CLOSE_PAREN:
            arg = self._eat_expression()
            result.add(arg)
            if self.peek(0).kind != SqTokenKind.COMMA:
                break
            self.advance()
        self.expect([SqTokenKind.CLOSE_PAREN])
        if len(result.children) == 0:
            result.add(None)
        return result

    def _eat_operator(self) -> Token:
        op = self.expect([
            SqTokenKind.PLUS,
            SqTokenKind.MINUS,
            SqTokenKind.STAR,
            SqTokenKind.SLASH,
            SqTokenKind.MODULUS,

            SqTokenKind.EQUALS,
            SqTokenKind.NOT_EQUALS,
            SqTokenKind.GREATER,
            SqTokenKind.GREATEREQ,
            SqTokenKind.LESS,
            SqTokenKind.LESSEQ,

            SqTokenKind.AND,
            SqTokenKind.OR,

            SqTokenKind.DOT,

            SqTokenKind.NOT,
        ])
        match op.kind:
            case SqTokenKind.PLUS | SqTokenKind.MINUS | SqTokenKind.STAR | SqTokenKind.SLASH | SqTokenKind.MODULUS \
                | SqTokenKind.EQUALS | SqTokenKind.NOT_EQUALS \
                | SqTokenKind.GREATER | SqTokenKind.GREATEREQ | SqTokenKind.LESS | SqTokenKind.LESSEQ \
                | SqTokenKind.AND | SqTokenKind.OR \
                | SqTokenKind.DOT:
                return Token(op.kind.name.lower(), op.kind.value)
            case _:
                raise NotImplementedError(op.kind)

    def _handle_block(self, *, include_open: bool) -> Tree:
        if include_open:
            self.expect([SqTokenKind.OPEN_CURLY])
        res = Tree(
            "block_expression",
            children=self._handle_stmt_list([SqTokenKind.CLOSE_CURLY])  # type: ignore (list covariance)
        )  
        self.expect([SqTokenKind.CLOSE_CURLY])
        return res

    def _descend_atom_expr(self) -> Tree:
        x = self.expect([
            SqTokenKind.INTEGER,
            SqTokenKind.DECIMAL,
            SqTokenKind.STRING,
            SqTokenKind.TRUE,
            SqTokenKind.FALSE,
            SqTokenKind.IDENTIFIER,
            SqTokenKind.OPEN_PAREN,
            SqTokenKind.OPEN_CURLY,
        ])

        res: Optional[Tree] = None

        match x.kind:
            case SqTokenKind.INTEGER | SqTokenKind.DECIMAL:
                name = x.kind.name.lower()
                res = Tree(name, children=[Token(name, x.literal)])
            case SqTokenKind.STRING:
                name = x.kind.name.lower()
                res = Tree(name, children=[Token(name, x.literal[1:-1])])
            case SqTokenKind.TRUE | SqTokenKind.FALSE:
                res = Tree("bool", children=[Token(x.literal, x.literal)])
            case SqTokenKind.IDENTIFIER:
                res = self._make_identifier(x)
            case SqTokenKind.OPEN_PAREN:
                res = Tree("grouped_expression", children=[self._eat_expression()])
                self.expect([SqTokenKind.CLOSE_PAREN])
            case SqTokenKind.OPEN_CURLY:
                res = self._handle_block(include_open=False)
            case _:
                raise NotImplementedError(x.kind)
        if res is None:  # type: ignore (though this will never run it may run in the future)
            raise NotImplementedError()
        res = self._wrap_expr(res)

        if self.peek(0).kind == SqTokenKind.OPEN_PAREN:
            args = self._eat_function_args()
            lhs = res
            res = Tree("fn_call_expr", children=[lhs, args])
        return self._wrap_expr(res)

    def _descend_unary_expr(self) -> Tree:
        expr_or_op = self.peek(0)
        if expr_or_op.kind in [SqTokenKind.MINUS, SqTokenKind.NOT]:
            op = self._eat_operator()
            rhs = self._descend_unary_expr()
            return self._wrap_expr(Tree("unary_expression", children=[op, rhs]))
        else:
            return self._descend_atom_expr()

    def _descend_dot_expr(self) -> Tree:
        expr = self._descend_atom_expr()
        while self.peek(0).kind == SqTokenKind.DOT:
            op = self._eat_operator()
            rhs = self._eat_identifier()

            expr = self._wrap_expr(Tree("binary_expression", children=[expr, op, rhs]))

            if self.peek(0).kind == SqTokenKind.OPEN_PAREN:
                args = self._eat_function_args()
                expr = self._wrap_expr(Tree("fn_call_expr", children=[expr, args]))
        return expr

    def _descend_mul_expr(self) -> Tree:
        expr = self._descend_dot_expr()
        while self.peek(0).kind in [SqTokenKind.STAR, SqTokenKind.SLASH, SqTokenKind.MODULUS]:
            op = self._eat_operator()
            rhs = self._descend_dot_expr()
            expr = self._wrap_expr(Tree("binary_expression", children=[expr, op, rhs]))        
        return expr


    def _descend_plus_expr(self) -> Tree:
        expr = self._descend_mul_expr()
        while self.peek(0).kind in [SqTokenKind.PLUS, SqTokenKind.MINUS]:
            op = self._eat_operator()
            rhs = self._descend_mul_expr()
            expr = self._wrap_expr(Tree("binary_expression", children=[expr, op, rhs])) 
        return expr

    def _descend_cmp_expr(self) -> Tree:
        expr = self._descend_plus_expr()
        while self.peek(0).kind in [SqTokenKind.EQUALS, SqTokenKind.NOT_EQUALS, SqTokenKind.GREATER, SqTokenKind.GREATEREQ, SqTokenKind.LESS, SqTokenKind.LESSEQ]:
            op = self._eat_operator()
            rhs = self._descend_plus_expr()
            expr = self._wrap_expr(Tree("binary_expression", children=[expr, op, rhs]))
        return expr
    
    def _descend_and_expr(self) -> Tree:
        expr = self._descend_cmp_expr()
        while self.peek(0).kind in [SqTokenKind.AND, SqTokenKind.OR]:
            op = self._eat_operator()
            rhs = self._descend_cmp_expr()
            expr = self._wrap_expr(Tree("binary_expression", children=[expr, op, rhs]))
        return expr

    def _descend_binary_expr(self) -> Tree:
        return self._descend_and_expr()

    def _eat_expression(self) -> Tree:
        return self._descend_binary_expr()


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

        if self.peek(0).kind == SqTokenKind.COLON:
            res.add(self._eat_user_type())

        self.expect([SqTokenKind.EQ])
        res.add(self._eat_expression())
        return res
    
    def _eat_function_definition_args(self) -> Tree:
        self.expect([SqTokenKind.OPEN_PAREN])
        result = Tree("fn_definition_args")
        while self.peek(0).kind != SqTokenKind.CLOSE_PAREN:
            ident = self._eat_identifier()
            typ = self._eat_user_type()
            result.add(Tree("fn_definition_arg", children=[ident, typ]))
            if self.peek(0).kind != SqTokenKind.COMMA:
                break
            self.advance()
        self.expect([SqTokenKind.CLOSE_PAREN])
        return result
    
    def _eat_function_definition(self) -> Tree:
        self.expect([SqTokenKind.FN])
        result = Tree("fn_definition", children=[self._cur_pub])
        self._cur_pub = None
        result.add(self._eat_identifier())
        result.add(self._eat_function_definition_args())
        ret_type = None
        if self.peek(0).kind == SqTokenKind.ARROW:
            ret_type = self._eat_return_user_type()
        result.add(ret_type)
        result.add(self._handle_block(include_open=True))
        return result

    def _eat_struct(self) -> Tree:
        self.expect([SqTokenKind.STRUCT])
        result = Tree("struct_definition", children=[self._cur_decorator, self._cur_pub])
        self._cur_pub = None
        self._cur_decorator = None

        result.add(self._eat_identifier())
        self.expect([SqTokenKind.OPEN_CURLY])
        
        field_list: list[Tree] = []
        cursor = self.peek(0)
        while cursor.kind != SqTokenKind.CLOSE_CURLY:
            field_pub: Optional[Tree] = None
            if self.peek(0).kind == SqTokenKind.PUB:
                self.advance()
                field_pub = Tree("PUB", children=[Token("PUB", "pub")])
            field_ident = self._eat_identifier()
            field_type = self._eat_user_type()
            field_list.append(Tree("struct_field", children=[field_pub, field_ident, field_type]))
            cursor = self.peek(0)
        if len(field_list) == 0:
            result.add(None)
        else:
            result.children.extend(field_list)
        self.expect([SqTokenKind.CLOSE_CURLY])

        return result
    
    def _eat_alias(self) -> Tree:
        self.expect([SqTokenKind.ALIAS])
        result = Tree("alias_definition")
        result.add(self._eat_identifier())
        self.expect([SqTokenKind.EQ])
        result.add(self._eat_identifier())
        return result
    
    def _eat_if_stmt(self) -> Tree:
        self.expect([SqTokenKind.IF])
        result = Tree("if_stmt")
        cond = self._eat_expression()
        result.add(cond)
        result.add(self._handle_block(include_open=True))
        if self.peek(0).kind == SqTokenKind.ELSE:
            self.advance()
            result.add(self._handle_block(include_open=True))
        else:
            result.add(None)
        return result
    
    def _eat_import(self) -> Tree:
        self.expect([SqTokenKind.IMPORT])
        result = Tree("import_stmt", children=[self._eat_identifier()])
        return result

    def _eat_from_import(self) -> Tree:
        self.expect([SqTokenKind.FROM])
        module_ident = self._eat_identifier()
        self.expect([SqTokenKind.IMPORT])
        cursor = self.expect([SqTokenKind.OPEN_SQUARE, SqTokenKind.STAR])
        if cursor.kind == SqTokenKind.OPEN_SQUARE:
            node: Optional[Tree] = None
            if self.peek(0).kind != SqTokenKind.CLOSE_SQUARE:
                ident_list = Tree("import_list")
                while True:
                    ident = self._eat_identifier()
                    ident_list.add(ident)
                    if self.peek(0).kind != SqTokenKind.COMMA:
                        break
                    self.advance()
                node = Tree("import_from_stmt", children=[module_ident, ident_list])
            else:
                node = Tree("import_from_stmt", children=[module_ident, None])
            self.expect([SqTokenKind.CLOSE_SQUARE])
            return node
        else:
            if cursor.kind != SqTokenKind.STAR:
                raise NotImplementedError(cursor.kind)
            return Tree("import_all_from_stmt", children=[module_ident])

    def _eat_while_stmt(self) -> Tree:
        self.expect([SqTokenKind.WHILE])
        cond = self._eat_expression()
        body = self._handle_block(include_open=True)
        return Tree("while_stmt", children=[cond, body])

    def _eat_statement(self) -> Optional[Tree]:
        tok = self.peek(0)
        result = Tree("statement")

        if self._cur_decorator != None and tok.kind not in [SqTokenKind.FN, SqTokenKind.STRUCT, SqTokenKind.PUB]:
            raise ParserError(f"Invalid token for decorator: {tok.kind}")
        if self._cur_pub != None and tok.kind not in [SqTokenKind.FN, SqTokenKind.STRUCT]:
            raise ParserError(f"Invalid token for pub: {tok.kind}")

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
            case SqTokenKind.FN:
                result.add(self._eat_function_definition())
            case SqTokenKind.STRUCT:
                result.add(self._eat_struct())
            case SqTokenKind.ALIAS:
                result.add(self._eat_alias())
            case SqTokenKind.LET:
                if self._cur_decorator != None:
                    raise ParserError("Decorators aren't allowed for let statements")
                result.add(self._eat_let())
            case SqTokenKind.RETURN:
                self.advance()
                prev_skip = self._skip_newlines
                self._skip_newlines = False
                peek_tok = self.peek(0)
                self._skip_newlines = prev_skip
                if peek_tok.kind == SqTokenKind.NEWLINE:
                    result.add(Tree("return_stmt", children=[None]))
                else:
                    result.add(Tree("return_stmt", children=[self._eat_expression()]))
            case SqTokenKind.IF:
                result.add(self._eat_if_stmt())
            case SqTokenKind.IMPORT:
                result.add(self._eat_import())
            case SqTokenKind.FROM:
                result.add(self._eat_from_import())
            case SqTokenKind.WHILE:
                result.add(self._eat_while_stmt())
            case SqTokenKind.BREAK:
                result.add(Tree("break_stmt"))
                self.advance()
            case SqTokenKind.CONTINUE:
                result.add(Tree("continue_stmt"))
                self.advance()
            case _:
                expr = self._eat_expression()

                if self.peek(0).kind == SqTokenKind.EQ:
                    self.advance()
                    rhs = self._eat_expression()
                    result.add(Tree("assignment", children=[expr, rhs]))
                else:
                    result.add(expr)
        if len(result.children) == 0:
            raise SerqInternalError()
        return result
    
    def _handle_stmt_list(self, until: list[SqTokenKind]) -> list[Tree]:
        result: list[Tree] = []
        tok = self.peek(0)
        while tok.kind != SqTokenKind.EOF and tok.kind not in until:
            stmt = self._eat_statement()
            if stmt != None:
                result.append(stmt)
            tok = self.peek(0)
        return result

    def parse(self) -> Tree:
        result = Tree("start", children=self._handle_stmt_list([]))  # type: ignore (list covariance)
        tok = self.peek(0)
        while tok.kind != SqTokenKind.EOF:
            stmt = self._eat_statement()
            if stmt != None:
                result.add(stmt)
            tok = self.peek(0)
        return result
