import pytest

reserved_keywords = [
    "and",
    "break",
    "continue",
    "else",
    "fn",
    "if",
    "let",
    "mut",
    "not",
    "or",
    "return",
    "while",
]

@pytest.mark.parametrize("keyword", reserved_keywords)
def test_let_reserved_keywords(executor, keyword):
    with pytest.raises(ValueError):
        executor(f"let {keyword} = 1")

@pytest.mark.parametrize("keyword", reserved_keywords)
def test_fn_name_reserved_keywords(executor, keyword):
    with pytest.raises(ValueError):
        code = f"""
        fn {keyword}() {{
            return
        }}"""
        executor(code)

@pytest.mark.parametrize("keyword", reserved_keywords)
def test_fn_args_reserved_keywords(executor, keyword):
    with pytest.raises(ValueError):
        code = f"""
        fn test({keyword}: int) {{
            return
        }}"""
        executor(code)