def test_empty_literal(capture_first_debug):
    assert capture_first_debug("dbg(\"\")") == ""


def test_quote_escape(capture_first_debug):
    assert capture_first_debug('dbg("\\"")') == '"'
