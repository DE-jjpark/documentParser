"""Azure Document Intelligence 클라이언트 (prebuilt-layout 모델).

필요 환경변수:
  AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT   예) https://<리소스>.cognitiveservices.azure.com
  AZURE_DOCUMENT_INTELLIGENCE_KEY

아래 필드명(``paragraphs``, ``content``, ``bounding_regions``, ``polygon``)은
문서만 보고 짐작한 게 아니라 azure-ai-documentintelligence==1.0.2를 실제
설치해서 모델 클래스를 직접 열어보고 확인했다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from document_parser.core.exceptions import MissingDependencyError


@dataclass
class LayoutParagraph:
    text: str
    # 하나의 단락(paragraph)이 여러 bounding_regions을 가질 수 있어(예: 단락이
    # 컬럼을 넘어가며 이어지는 경우) 단일 bbox가 아니라 리스트로 담는다.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class AzureLayoutResult:
    paragraphs: list[LayoutParagraph]


class AzureDocumentIntelligenceClient:
    def __init__(self) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise MissingDependencyError(
                "Azure Document Intelligence support requires the 'azure' extra: "
                "pip install 'document-parser[azure]'"
            ) from exc

        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise MissingDependencyError(
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT / AZURE_DOCUMENT_INTELLIGENCE_KEY "
                "environment variables are required"
            )

        self._client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def analyze_layout(self, image_bytes: bytes) -> AzureLayoutResult:
        """페이지 이미지 1장을 통째로 보낸다(다이어그램 메모: 크롭 없이 전체
        페이지 — 어차피 페이지 단위 과금)."""
        poller = self._client.begin_analyze_document("prebuilt-layout", body=image_bytes)
        result = poller.result()
        paragraphs = [
            LayoutParagraph(text=p.content, bboxes=_polygon_to_bboxes(p.bounding_regions))
            for p in (result.paragraphs or [])
        ]
        return AzureLayoutResult(paragraphs=paragraphs)


def _polygon_to_bboxes(bounding_regions) -> list[tuple[float, float, float, float]]:
    """각 bounding region의 폴리곤(윗왼쪽부터 시계방향 [x0,y0,x1,y1,...] 평탄화
    좌표) 하나하나를 축정렬 bbox로 변환 — region이 여러 개면 리스트로 그대로 반환.

    TODO: 실제 in4u Azure 엔드포인트 기준 좌표 단위 확인 필요 — 이미지 입력이면
    보낸 이미지 픽셀 좌표와 같아야 하는데, 아직 실제 호출로 검증한 적은 없다.
    """
    boxes = []
    for region in bounding_regions or []:
        xs = region.polygon[0::2]
        ys = region.polygon[1::2]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes
