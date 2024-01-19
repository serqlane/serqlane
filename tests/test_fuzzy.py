import pytest
from contextlib import suppress

from serqlane.astcompiler import ModuleGraph, SerqTypeInferError
from serqlane.common import SerqInternalError
from serqlane.parser import ParserError


try:
    from lark import Lark
    import hypothesis
    from hypothesis.extra.lark import from_lark
except ImportError:
    pass
else:
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
