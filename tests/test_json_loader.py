import pytest

from document_parser import ElementType, ParsingEngine, ParsingFailedError


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def test_flattens_nested_dict_and_list(engine):
    data = b'{"name": "doc", "tags": ["a", "b"], "meta": {"pages": 3}}'
    document = engine.parse("doc.json", data=data)

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.TEXT
    lines = element.text.split("\n")
    assert "name: doc" in lines
    assert "tags[0]: a" in lines
    assert "tags[1]: b" in lines
    assert "meta.pages: 3" in lines


def test_empty_json_object_returns_no_elements(engine):
    document = engine.parse("empty.json", data=b"{}")
    assert document.elements == []


def test_invalid_json_raises_parsing_failed_error(engine):
    with pytest.raises(ParsingFailedError):
        engine.parse("bad.json", data=b"{not valid json")
