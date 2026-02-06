from sprig.utils.text import extract_entities_regex, tokenize


def test_tokenize_basic():
    assert tokenize("Hello, world!") == ["hello", "world"]


def test_extract_entities_regex():
    text = "Barack Obama visited New York City." 
    ents = extract_entities_regex(text)
    assert "Barack Obama" in ents
    assert "New York City" in ents
