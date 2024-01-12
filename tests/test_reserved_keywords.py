import pytest
from serqlane.astcompiler import RESERVED_KEYWORDS
from serqlane.new_parser import ParserError

@pytest.mark.parametrize("keyword", RESERVED_KEYWORDS)
def test_let_reserved_keywords(executor, keyword):
    with pytest.raises(ParserError):
        executor(f"let {keyword} = 1")

@pytest.mark.parametrize("keyword", RESERVED_KEYWORDS)
def test_fn_name_reserved_keywords(executor, keyword):
    with pytest.raises(ParserError):
        code = f"""
        fn {keyword}() {{
            return
        }}"""
        executor(code)

@pytest.mark.parametrize("keyword", RESERVED_KEYWORDS)
def test_fn_args_reserved_keywords(executor, keyword):
    with pytest.raises(ParserError):
        code = f"""
        fn test({keyword}: int) {{
            return
        }}"""
        executor(code)

@pytest.mark.parametrize("keyword", RESERVED_KEYWORDS)
def test_struct_name_reserved_keywords(executor, keyword):
    with pytest.raises(ParserError):
        code = f"""
        struct {keyword} {{
            a: int
        }}"""
        executor(code)