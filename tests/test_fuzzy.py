import hypothesis
import pytest
from hypothesis.extra.lark import from_lark
from lark import Lark
from contextlib import suppress

from serqlane.astcompiler import ModuleGraph, SerqTypeInferError
from serqlane.common import SerqInternalError
from serqlane.parser import ParserError


with open("serqlane/grammar.lark") as fp:
    grammar = fp.read()


lark_parser = Lark(grammar)


@pytest.mark.fuzz
@hypothesis.settings()
@hypothesis.given(
    from_lark(lark_parser)
)
def test_fuzz_compiler(code: str):
    graph = ModuleGraph()

    with suppress(ValueError, AssertionError, ParserError, SerqInternalError, SerqTypeInferError):
        graph.load("<string>", code)
