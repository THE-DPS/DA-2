from sentiment_nlp.preprocessing import normalize_text


def test_normalize_html_and_whitespace():
    assert normalize_text("  Great&nbsp;<b>screen</b> \n life ") == "Great screen life"


def test_preserves_negation_and_punctuation():
    result = normalize_text("This is NOT good!!!")
    assert "NOT" in result
    assert "!!!" in result


def test_unicode_normalization_and_type_check():
    assert normalize_text("cafe\u0301") == "café"
    try:
        normalize_text(None)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-string input")
