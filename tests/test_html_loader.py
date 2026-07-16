import pytest

from document_parser import ElementType, ParsingEngine


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def test_headings_paragraphs_and_list(engine):
    data = b"""
    <html><body>
      <h1>Title</h1>
      <p>First <b>paragraph</b> with inline markup.</p>
      <h2>Section</h2>
      <ul><li>one</li><li>two</li></ul>
    </body></html>
    """
    document = engine.parse("doc.html", data=data)

    assert [el.type for el in document.elements] == [
        ElementType.HEADING,
        ElementType.TEXT,
        ElementType.HEADING,
        ElementType.LIST,
        ElementType.LIST,
    ]
    assert document.elements[0].text == "Title"
    assert document.elements[0].metadata["level"] == 1
    assert document.elements[1].text == "First paragraph with inline markup."
    assert document.elements[2].metadata["level"] == 2
    assert document.elements[3].text == "one"
    assert document.elements[4].text == "two"


def test_table_flattened_to_tab_separated_rows(engine):
    data = b"""
    <table>
      <tr><td>Name</td><td>Score</td></tr>
      <tr><td>Alice</td><td>90</td></tr>
    </table>
    """
    document = engine.parse("table.html", data=data)

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.TABLE
    assert element.text == "Name\tScore\nAlice\t90"


def test_htm_extension_uses_same_loader(engine):
    document = engine.parse("doc.htm", data=b"<h1>Hi</h1>")
    assert document.elements[0].type == ElementType.HEADING
    assert document.elements[0].text == "Hi"
