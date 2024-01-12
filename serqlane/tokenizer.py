import pathlib

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Iterator

from serqlane.common import SerqInternalError



class TokenizerError(Exception): ...



class SqTokenKind(Enum):
    ERROR = "<error>"
    EOF = "<eof>"
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
    ELSE = "else"

    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"

    IMPORT = "import"
    FROM = "from"

    STRUCT = "struct"
    FN = "fn"
    ALIAS = "alias"

    # special
    IDENTIFIER = "{identifier}"
    INTEGER = "{integer}"
    DECIMAL = "{decimal}"
    STRING = "{string}"
    SINGLE_LINE_DOC_COMMENT = "{doc}"

    def is_keyword(self) -> bool:
        return not self.is_symbolic() and self.name.isalnum()

    def is_symbolic(self) -> bool:
        if self in [SqTokenKind.ERROR, SqTokenKind.EOF, SqTokenKind.NEWLINE]:
            return False
        return not self.value[0].isalnum() and not (self.value.startswith("{") and self.value.endswith("}"))

KEYWORDS = dict(
    (x.value, x) for x in SqTokenKind if x.is_keyword()
)

# Also contains () and similar
SINGLECHAR_OPERATORS = dict(
    (x.value, x) for x in SqTokenKind if len(x.value) == 1 and x.is_symbolic()
)
MULTICHAR_MARKERS = frozenset(
    x.value[0] for x in SqTokenKind if len(x.value) > 1 and x.is_symbolic()
)
MULTICHAR_OPERATORS = dict(
    (x.value, x) for x in SqTokenKind if len(x.value) > 1 and x.is_symbolic()
)
for t in SqTokenKind:
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


class SqToken:
    def __init__(self, kind: SqTokenKind, literal: str, *, line: int, column_start: int, column_end: int) -> None:
        self.kind = kind
        self.literal = literal

        self.line = line
        self.column_start = column_start
        self.column_end = column_end

    def get_line_info(self) -> LineInfo:
        return LineInfo(
            self.line,
            self.column_start,
            self.column_end,
        )
 
    def render(self) -> str:
        val = self.kind.value
        if val.startswith("{"):
            match self.kind:
                case SqTokenKind.IDENTIFIER:
                    return val.format(identifier=self.literal)
                case SqTokenKind.INTEGER:
                    return val.format(integer=self.literal)
                case SqTokenKind.DECIMAL:
                    return val.format(decimal=self.literal)
                case SqTokenKind.STRING:
                    return val.format(string=self.literal)
                case SqTokenKind.SINGLE_LINE_DOC_COMMENT:
                    return val.format(doc=self.literal)
                case _:
                    raise SerqInternalError(f"Unhandled token kind: {self.kind.name}")
        else:
            return val


class Tokenizer:
    def __init__(self, data: str, *, filepath: Optional[pathlib.Path] = None) -> None:
        self.data = data.strip()
        self.offset = 0
        self.line = 0
        self.column = 0
        self.filepath = filepath
        self.remaining = len(self.data)

    def advance(self) -> None:
        c = self.data[self.offset]
        self.offset += 1
        self.remaining -= 1
        if c == "\n":
            self.line += 1
            self.column = 0
        else:
            self.column += 1

    def peek(self, offset=1) -> str:
        if self.remaining > offset:
            return self.data[self.offset + offset]
        return ""
    
    def take_while(self, pred: Callable[[str], bool]) -> str:
        c = self.peek(0)
        if not pred(c):
            return ""
        result = c
        if self.remaining <= 0:
            return result
        self.advance()
        while self.remaining > 0:
            c = self.peek(0)
            if not pred(c):
                break
            result += c
            self.advance()
        return result

    def make_token(self, kind: SqTokenKind, literal: str) -> SqToken:
        return SqToken(
            kind=kind,
            literal=literal,
            line=self.line,
            column_start=self.column - len(literal),
            column_end=self.column,
        )

    def report_error(self, message: str, tok: SqToken):
        if self.filepath != None:
            location_str = f"\nAt: {self.filepath.absolute()}:{tok.get_line_info().to_path_cursor()}"
        else:
            location_str = f"\nline: {tok.get_line_info().line}\ncolumn: {tok.get_line_info().column_start}"
        
        message += location_str
        raise TokenizerError(message)

    def process_iter(self) -> Iterator[SqToken]:
        while self.remaining > 0:
            c = self.peek(0)

            if c.isalpha():
                # identifier or keyword
                # manually inlined take_while
                ident = c
                self.advance()
                while self.remaining > 0:
                    cc = self.peek(0)
                    if not (cc.isalnum() or cc == "_"):
                        break
                    ident += cc
                    self.advance()
                kind = KEYWORDS.get(ident, SqTokenKind.IDENTIFIER)
                yield SqToken(
                    kind=kind,
                    literal=ident,
                    line=self.line,
                    column_start=self.column - len(ident),
                    column_end=self.column,
                )
            elif c.isspace():
                # space; advance keeps track of line and column already
                if c == "\n":
                    yield SqToken(
                        kind=SqTokenKind.NEWLINE,
                        literal=c,
                        line=self.line,
                        column_start=self.column - len(c),
                        column_end=self.column,
                    )
                self.advance()
            elif c.isnumeric():
                # int or float
                # manually inlined take_while
                lit = c
                self.advance()
                while self.remaining > 0:
                    cc = self.peek(0)
                    if not (cc.isnumeric() or cc == "."):
                        break
                    lit += cc
                    self.advance()
                dot_count = lit.count(".")
                if dot_count == 0:
                    yield SqToken(
                        kind=SqTokenKind.INTEGER,
                        literal=lit,
                        line=self.line,
                        column_start=self.column - len(lit),
                        column_end=self.column,
                    )
                elif dot_count == 1:
                    yield SqToken(
                        kind=SqTokenKind.DECIMAL,
                        literal=lit,
                        line=self.line,
                        column_start=self.column - len(lit),
                        column_end=self.column,
                    )
                else:
                    self.report_error(
                        f"Numeric literal with {dot_count} dots found: {lit}",
                        self.make_token(SqTokenKind.ERROR, lit)
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
                    else:
                        escaping = False
                    return True
                lit = c + self.take_while(pred)
                if not found_quote:
                    self.report_error(
                        f"Encountered an unclosed string literal: {lit}",
                        self.make_token(SqTokenKind.ERROR, lit)
                    )
                yield SqToken(
                    kind=SqTokenKind.STRING,
                    literal=lit,
                    line=self.line,
                    column_start=self.column - len(lit),
                    column_end=self.column,
                )
            else:
                # symbol or comment
                if c == "/" and self.peek(1) == "/":
                    # comment
                    com = self.take_while(lambda x: x != "\n")
                    if com.startswith("///"):
                        yield self.make_token(
                            SqTokenKind.SINGLE_LINE_DOC_COMMENT,
                            com,
                        )
                else:
                    # symbol
                    put_multichar = False
                    if c in MULTICHAR_MARKERS:
                        tmp = c + self.peek(1)
                        multichar_op = MULTICHAR_OPERATORS.get(tmp, None)
                        if multichar_op != None:
                            yield SqToken(
                                kind=multichar_op,
                                literal=tmp,
                                line=self.line,
                                column_start=self.column - len(tmp),
                                column_end=self.column,
                            )
                            self.advance()
                            put_multichar = True
                    if not put_multichar:
                        if c not in SINGLECHAR_OPERATORS:
                            self.report_error(f"Unrecognized symbol: '{c}'", self.make_token(SqTokenKind.ERROR, c))
                        yield SqToken(
                            kind=SINGLECHAR_OPERATORS[c],
                            literal=c,
                            line=self.line,
                            column_start=self.column - len(c),
                            column_end=self.column,
                        )
                    self.advance()
        yield self.make_token(SqTokenKind.EOF, "")

    def process(self) -> list[SqToken]:
        return list(self.process_iter())
