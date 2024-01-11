import pathlib

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from serqlane.common import SerqInternalError



class TokenizerError(Exception): ...



class TokenKind(Enum):
    ERROR = "<error>"
    NEWLINE = "<newline>"

    # symbols
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    EQ = "="
    MODULUS = "%"
    EQUALS = "=="
    NOT_EQUALS = "!="
    LESS = "<"
    LESSEQ = "<="
    GREATER = ">"
    GREATEREQ = ">="

    OPEN_PAREN = "("
    CLOSE_PAREN = ")"
    OPEN_CURLY = "{"
    CLOSE_CURLY = "}"
    OPEN_SQUARE = "["
    CLOSE_SQUARE = "]"

    ARROW = "->"
    COLON = ":"
    DOT = "."
    COMMA = ","
    AT = "@"

    # keywords
    TRUE = "true"
    FALSE = "false"
    
    AND = "and"
    OR = "or"
    
    LET = "let"
    MUT = "mut"
    PUB = "pub"

    WHILE = "while"
    IF = "if"

    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"

    IMPORT = "import"
    FROM = "from"

    # special
    IDENTIFIER = "{identifier}"
    INTEGER = "{integer}"
    DECIMAL = "{decimal}"
    STRING = "{string}"
    SINGLE_LINE_DOC_COMMENT = "{doc}"

    def is_symbolic(self) -> bool:
        if self in [TokenKind.ERROR, TokenKind.NEWLINE]:
            return False
        return not self.value[0].isalnum() and not (self.value.startswith("{") and not self.value.startswith("}"))

MULTICHAR_MARKERS = set(
    x.value[0] for x in TokenKind if len(x.value) > 1 and x.is_symbolic()
)
for t in TokenKind:
    if t.is_symbolic():
        # Some logic must change if operators become longer
        assert len(t.value) < 3, t.value


@dataclass
class LineInfo:
    line: int
    column_start: int
    column_end: int

    def to_path_selection(self) -> str:
        return f"{self.to_path_cursor()}-{self.column_end + 1}"

    def to_path_cursor(self) -> str:
        return f"{self.line + 1}:{self.column_start + 1}"


class Token:
    def __init__(self, kind: TokenKind, literal: str, line_info: LineInfo, *, value=None) -> None:
        self.kind = kind
        self.literal = literal
        self.line_info = line_info
        self.value = value
 
    def render(self) -> str:
        val = self.kind.value
        if val.startswith("{"):
            match self.kind:
                case TokenKind.IDENTIFIER:
                    return val.format(identifier=self.literal)
                case TokenKind.INTEGER:
                    return val.format(integer=str(self.value))
                case TokenKind.DECIMAL:
                    return val.format(decimal=str(self.value))
                case TokenKind.STRING:
                    return val.format(string=f"\"{self.value}\"")
                case TokenKind.SINGLE_LINE_DOC_COMMENT:
                    return val.format(doc=self.value)
                case _:
                    raise SerqInternalError(f"Unhandled token kind: {self.kind.name}")
        else:
            return val


class Tokenizer:
    def __init__(self, data: str, *, filepath: Optional[pathlib.Path] = None) -> None:
        self.data = data
        self.offset = 0
        self.line = 0
        self.column = 0
        self.filepath = filepath

    def remaining(self) -> int:
        return len(self.data) - self.offset

    def advance(self) -> None:
        c = self.data[self.offset]
        self.offset += 1
        if c == "\n":
            self.line += 1
            self.column = 0
        else:
            self.column += 1

    def peek(self, offset=1) -> str:
        if self.remaining() > offset:
            return self.data[self.offset + offset]
        return ""
    
    def take_while(self, pred: Callable[[str], bool]) -> str:
        c = self.peek(0)
        if not pred(c):
            return ""
        result = c
        self.advance()
        while self.remaining() > 0:
            c = self.peek(0)
            if not pred(c):
                break
            result += c
            self.advance()
        return result

    def make_token(self, kind: TokenKind, literal: str, *, value=None) -> Token:
        return Token(
            kind=kind,
            literal=literal,
            line_info=LineInfo(
                self.line,
                self.column - len(literal),
                self.column,
            ),
            value=value,
        )

    def report_error(self, message: str, tok: Token):
        if self.filepath != None:
            location_str = f"\nAt: {self.filepath.absolute()}:{tok.line_info.to_path_cursor()}"
        else:
            location_str = f"\nline: {tok.line_info.line}\ncolumn: {tok.line_info.column_start}"
        
        message += location_str
        raise TokenizerError(message)

    def run(self) -> list[Token]:
        result = []

        while True:
            c = self.peek(0)
            assert len(c) == 1

            if c.isnumeric():
                # int or float
                lit = self.take_while(lambda x: x.isnumeric() or x == ".")
                dot_count = lit.count(".")
                if dot_count == 0:
                    result.append(self.make_token(TokenKind.INTEGER, lit, value=int(lit)))
                elif dot_count == 1:
                    result.append(self.make_token(TokenKind.DECIMAL, lit, value=float(lit)))
                else:
                    self.report_error(
                        f"Numeric literal with {dot_count} dots found: {lit}",
                        self.make_token(TokenKind.ERROR, lit)
                    )
            elif c == '"':
                self.advance()
                escaping = False
                found_quote = False
                def pred(x: str):
                    nonlocal escaping
                    nonlocal found_quote
                    if found_quote:
                        return False
                    is_quote = x == '"' and not escaping
                    if is_quote:
                        found_quote = True
                    if x == "\\":
                        escaping = not escaping # handle self escape
                    return True
                lit = c + self.take_while(pred)
                if not found_quote:
                    self.report_error(
                        f"Encountered an unclosed string literal: {lit}",
                        self.make_token(TokenKind.ERROR, lit)
                    )
                result.append(self.make_token(TokenKind.STRING, lit, value=lit[1:-1]))
            elif c.isalpha():
                # identifier or keyword
                ident = self.take_while(lambda x: x.isalnum() or x == "_")
                kind = TokenKind(ident) if ident in TokenKind else TokenKind.IDENTIFIER
                result.append(self.make_token(kind, ident))
            elif c.isspace():
                # space; advance keeps track of line and column already
                if c == "\n":
                    result.append(self.make_token(TokenKind.NEWLINE, c))
            else:
                # symbol or comment
                if c == "/" and self.peek(1) == "/":
                    # comment
                    com = self.take_while(lambda x: x != "\n")
                    if com.startswith("///"):
                        result.append(self.make_token(
                            TokenKind.SINGLE_LINE_DOC_COMMENT,
                            com,
                            value=com[3:].strip()
                        ))
                else:
                    # symbol
                    full = c
                    if c in MULTICHAR_MARKERS:
                        tmp = c + self.peek(1)
                        if len(tmp) == 2 and tmp in TokenKind:
                            full = tmp
                            self.advance()
                    if full not in TokenKind:
                        self.report_error(f"Unrecognized symbol: '{full}'", self.make_token(TokenKind.ERROR, full))
                    result.append(self.make_token(TokenKind(full), full))

            self.advance()
            if self.remaining() <= 0:
                break

        return result


if __name__ == "__main__":
    code = """
let x = 2.1
"""
    p = pathlib.Path("test.serq")
    tokenizer = Tokenizer(p.read_text(), filepath=p)
    toks = tokenizer.run()
    print([x.render() for x in toks])
