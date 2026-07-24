"""layout.py의 ONNX(HPI) 병렬 경로 단위 테스트.

실제 HPI 플러그인(ultra-infer-python)은 linux 전용 wheel이라 로컬에서
설치·실행이 안 된다 -- _get_model_onnx()를 mock해서 폴백/정상 배선만
확인한다. 진짜 정확도 검증은 linux 배포 환경(Databricks)에서 진행.
"""

from unittest.mock import MagicMock, patch

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pdf extra not installed")

from document_parser.parsing.loaders.pdf.layout import (  # noqa: E402
    _page_layout_from_predict_result,
    analyze_page_onnx,
)


def _blank_page():
    doc = pymupdf.open()
    page = doc.new_page()
    return doc, page


def test_page_layout_from_predict_result_builds_boxes_and_detects_figures():
    result = {
        "boxes": [
            {"label": "text", "coordinate": [10, 10, 100, 50], "cls_id": 22, "order": 1},
            {"label": "table", "coordinate": [10, 60, 200, 150], "cls_id": 21, "order": 2},
        ]
    }

    layout = _page_layout_from_predict_result(result, has_text_layer=True)

    assert layout.has_text_layer is True
    assert layout.has_figures is True  # table은 _FIGURE_LABELS에 포함
    assert [b.label for b in layout.boxes] == ["text", "table"]


def test_page_layout_from_predict_result_no_figures_when_only_text():
    result = {"boxes": [{"label": "text", "coordinate": [0, 0, 10, 10], "cls_id": 22}]}

    layout = _page_layout_from_predict_result(result, has_text_layer=True)

    assert layout.has_figures is False


def test_analyze_page_onnx_falls_back_to_heuristic_when_dependency_error():
    """HPI 플러그인(ultra-infer-python) 미설치 시 paddlex가 던지는
    DependencyError -- ImportError가 아니라서 별도로 잡아야 한다."""
    deps = pytest.importorskip("paddlex.utils.deps", reason="layout extra not installed")
    DependencyError = deps.DependencyError

    doc, page = _blank_page()
    try:
        with patch(
            "document_parser.parsing.loaders.pdf.layout._get_model_onnx",
            side_effect=DependencyError("plugin not installed"),
        ):
            layout = analyze_page_onnx(page)
        assert layout.boxes == []
        assert layout.has_text_layer is False
    finally:
        doc.close()


def test_analyze_page_onnx_falls_back_to_heuristic_when_import_error():
    doc, page = _blank_page()
    try:
        with patch(
            "document_parser.parsing.loaders.pdf.layout._get_model_onnx",
            side_effect=ImportError("no paddlex"),
        ):
            layout = analyze_page_onnx(page)
        assert layout.has_text_layer is False
    finally:
        doc.close()


def test_analyze_page_onnx_uses_model_result_when_available():
    doc, page = _blank_page()
    try:
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            {
                "boxes": [
                    {"label": "text", "coordinate": [0, 0, 10, 10], "cls_id": 22, "order": 1}
                ]
            }
        ]
        with patch(
            "document_parser.parsing.loaders.pdf.layout._get_model_onnx",
            return_value=mock_model,
        ):
            layout = analyze_page_onnx(page)
        assert len(layout.boxes) == 1
        assert layout.boxes[0].label == "text"
        mock_model.predict.assert_called_once()
    finally:
        doc.close()
