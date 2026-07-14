"""Tests for the pdf loader's per-page routing (native vs AzureDI+VLM stub path).

Builds tiny synthetic PDFs with pymupdf itself rather than shipping fixture
files, mirroring the "hello world" style already used for text loader tests.
"""

import pymupdf
import pytest

from document_parser import ElementType, ParsingEngine
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
    document = engine.parse("doc.pdf", data=_pdf_with_image())

    sources = {el.metadata.get("source") for el in document.elements}
    assert sources == {"azure_di", "vlm"}

    vlm_elements = [el for el in document.elements if el.metadata.get("source") == "vlm"]
    assert len(vlm_elements) == 1
    assert vlm_elements[0].type == ElementType.IMAGE
    assert vlm_elements[0].bbox is not None


def test_needs_heavy_path_routing_rule():
    assert needs_heavy_path(PageLayout(has_figures=True, has_text_layer=True)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=False)) is True
    assert needs_heavy_path(PageLayout(has_figures=False, has_text_layer=True)) is False
