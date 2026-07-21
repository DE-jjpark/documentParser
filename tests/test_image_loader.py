"""단독 이미지(png/jpg/jpeg/webp) 로더 테스트.

실제 VLM 호출은 자격증명이 없는 환경에서도 테스트가 돌아야 하므로
document_parser.parsing.loaders.image.get_client를 mock한다 — pdf
로더 테스트(test_pdf_loader.py)가 caption_figures를 mock하는 것과 같은 패턴.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from document_parser import ElementType, ParsingEngine
from document_parser.parsing.clients.vlm import VLMCaptionResult


@pytest.fixture(scope="module")
def engine() -> ParsingEngine:
    return ParsingEngine()


def _fake_client(response_text: str, usage: dict | None = None) -> MagicMock:
    client = MagicMock()
    client.caption_image.return_value = VLMCaptionResult(text=response_text, usage=usage)
    return client


@pytest.mark.parametrize("fmt", ["png", "jpg", "jpeg", "webp"])
def test_image_formats_route_through_vlm(engine, fmt):
    response = "[CONTENT]\n갈색 강아지 사진입니다.\n[SUMMARY]\n강아지 한 마리가 있는 사진."
    with patch(
        "document_parser.parsing.loaders.image.get_client",
        return_value=_fake_client(response),
    ):
        document = engine.parse(f"photo.{fmt}", data=b"fake-image-bytes")

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.IMAGE
    assert element.text == "갈색 강아지 사진입니다."
    assert element.summary == "강아지 한 마리가 있는 사진."
    assert element.metadata["source"] == "vlm"


def test_image_loader_fast_tier_skips_vlm_entirely(engine):
    """이미지 로더는 VLM이 유일한 콘텐츠 출처라 tier="fast"면 호출 자체를
    안 하고, "감지는 했지만 캡션 안 만듦"을 명시하는 빈 요소만 돌려준다."""
    client = _fake_client("[CONTENT]\n설명\n[SUMMARY]\n요약")
    with patch(
        "document_parser.parsing.loaders.image.get_client",
        return_value=client,
    ):
        document = engine.parse("photo.png", data=b"fake-image-bytes", tier="fast")

    client.caption_image.assert_not_called()
    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.IMAGE
    assert element.text == ""
    assert element.metadata["source"] == "skipped_fast_tier"


def test_image_loader_passes_correct_mime_type_per_extension(engine):
    response = "[CONTENT]\n설명\n[SUMMARY]\n요약"
    client = _fake_client(response)
    with patch(
        "document_parser.parsing.loaders.image.get_client",
        return_value=client,
    ):
        engine.parse("photo.webp", data=b"fake-image-bytes")

    client.caption_image.assert_called_once()
    _, _, mime_type = client.caption_image.call_args[0]
    assert mime_type == "image/webp"


def test_image_loader_extracts_mermaid_from_flowchart_response(engine):
    response = "[CONTENT]\n```mermaid\ngraph TD;\nA-->B;\n```\n[SUMMARY]\n순서도 설명."
    with patch(
        "document_parser.parsing.loaders.image.get_client",
        return_value=_fake_client(response),
    ):
        document = engine.parse("diagram.png", data=b"fake-image-bytes")

    element = document.elements[0]
    assert element.text == "graph TD;\nA-->B;"
    assert element.metadata["mermaid"] == "graph TD;\nA-->B;"
    assert element.summary == "순서도 설명."


def test_image_loader_captures_vlm_usage(engine):
    response = "[CONTENT]\n설명\n[SUMMARY]\n요약"
    usage = {"total_tokens": 42}
    with patch(
        "document_parser.parsing.loaders.image.get_client",
        return_value=_fake_client(response, usage=usage),
    ):
        document = engine.parse("photo.jpg", data=b"fake-image-bytes")

    assert document.elements[0].metadata["vlm_usage"] == usage


requires_real_vlm = pytest.mark.skipif(
    not (os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN")),
    reason="DATABRICKS_HOST / DATABRICKS_TOKEN 환경변수 필요 (실제 in4u AI Gateway 호출)",
)


@requires_real_vlm
def test_image_loader_real_call(engine):
    """실제 in4u Databricks AI Gateway로 png 한 장을 캡션 요청해서 파이프라인
    전체(로더 등록 -> VLM 호출 -> content/summary 분리)가 살아있는지 확인."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 100), color="white").save(buf, format="PNG")

    document = engine.parse("photo.png", data=buf.getvalue())

    assert len(document.elements) == 1
    element = document.elements[0]
    assert element.type == ElementType.IMAGE
    assert element.text
