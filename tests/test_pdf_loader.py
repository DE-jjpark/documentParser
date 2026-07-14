"""Tests for the pdf loader's per-page routing (native vs AzureDI+VLM path).

Builds tiny synthetic PDFs with pymupdf itself rather than shipping fixture
files, mirroring the "hello world" style already used for text loader tests.

layout.analyze_page() itself is exercised for real (against whatever's
installed: real PP-DocLayoutV2 if the 'layout' extra + weights are present,
otherwise the pymupdf heuristic fallback -- both correctly classify these
synthetic pages). azure_di/vlm are mocked at the pdf-loader-import boundary
since they need real Azure credentials this environment doesn't have.
"""

from unittest.mock import patch

import pymupdf
import pytest

from document_parser import ElementType, ParsingEngine
from document_parser.core.models import BBox, DocumentElement
from document_parser.parsing.loaders.pdf.layout import PageLayout, needs_heavy_path


def _text_only_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello world")
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_with_image() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "a caption above a figure")
    pix = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (255, 0, 0))
    page.insert_image(pymupdf.Rect(72, 200, 172, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def test_text_only_page_takes_native_route(engine):
    document = engine.parse("doc.pdf", data=_text_only_pdf())

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.TEXT
    assert element.text == "hello world"
    assert element.page == 1
    assert element.bbox is not None
    assert element.metadata["source"] == "native"


def test_page_with_figure_takes_azure_di_and_vlm_route(engine):
    fake_azure_element = DocumentElement(
        type=ElementType.TEXT, text="from azure", page=1, metadata={"source": "azure_di"}
    )
    fake_vlm_element = DocumentElement(
        type=ElementType.IMAGE,
        text="from vlm",
        page=1,
        bbox=BBox(x0=0, y0=0, x1=1, y1=1),
        metadata={"source": "vlm"},
    )

    with (
        patch(
            "document_parser.parsing.loaders.pdf.extract_with_azure_di",
            return_value=[fake_azure_element],
        ) as mock_azure,
        patch(
            "document_parser.parsing.loaders.pdf.caption_figures",
            return_value=[fake_vlm_element],
        ) as mock_vlm,
    ):
        document = engine.parse("doc.pdf", data=_pdf_with_image())

    mock_azure.assert_called_once()
    mock_vlm.assert_called_once()
    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"azure_di", "vlm"}


def test_needs_heavy_path_routing_rule():
    assert needs_heavy_path(PageLayout(has_figures=True, has_text_layer=True)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=False)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=True)) is False


def test_azure_di_and_vlm_are_placeholders_for_now(engine):
    """azure_di/vlm real SDK calls are intentionally disabled right now (no
    usable in4u credentials yet, and VLM calls cost money per image) -- both
    return a fixed placeholder instead, so parsing doesn't fail and chunking
    still has text to work with. No mocking here: this is the actual current
    behavior, not a stand-in for a real call."""
    document = engine.parse("doc.pdf", data=_pdf_with_image())

    azure_elements = [el for el in document.elements if el.metadata.get("source") == "azure_di"]
    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]

    assert len(azure_elements) == 1
    assert azure_elements[0].metadata.get("stub") is True
    assert azure_elements[0].text

    assert len(vlm_elements) == 1
    assert vlm_elements[0].metadata.get("stub") is True
    assert vlm_elements[0].text
    assert vlm_elements[0].bbox is not None
