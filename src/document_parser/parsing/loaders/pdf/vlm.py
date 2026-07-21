"""VLM 그림/표 캡션·구조 추출 경로.

다이어그램 메모: AzureDI와 달리 먼저 크롭한 다음 VLM에 보낸다 — 페이지
전체보다 타이트하게 자른 영역이 정확도가 더 좋다.

실제 in4u Databricks AI Gateway(Claude Sonnet 4.6)로 연동 확인함 —
parsing.clients.vlm.VLMClient 참고. 크롭은 pymupdf의 get_pixmap(clip=...)로
하는데, clip은 포인트 좌표를 받으므로(픽셀 변환 불필요) box.bbox를 그대로
쓴다 — layout.py/azure_di.py에서 모델이 "돌려주는" 좌표를 변환해야 했던 것과는
반대 방향이라 혼동하지 말 것.

content/summary 분리: VLM한테 "[CONTENT]"/"[SUMMARY]" 두 구획으로 나눠서
답하게 하고(vlm_caption.split_content_summary), DocumentElement.text에는
content를, .summary에는 summary를 담는다 — "내용 그 자체"와 "요약"을 별개로
원한다는 요청에 따른 것.
  - 표(table): content = 마크다운 표(문법 그대로), summary = 2~4문장 요약.
    다만 AzureDI가 그 표를 찾았으면(graph.py의 merge 노드) content는 DI의
    실제 구조에서 뽑은 마크다운으로 덮어써진다 — VLM의 마크다운은 DI가 못
    찾았을 때만 그대로 남는 폴백이다.
  - 이미지(image/chart/순서도/서명 등): content는 종류별로 다르다 —
    순서도·다이어그램이면 Mermaid 코드(그 자체가 text가 됨, metadata["mermaid"]
    에도 같은 걸 남겨 명시적으로 판별 가능하게 함), 서명·손글씨면 실제로
    쓰인 글자 그대로(OCR), 그 외엔 평범한 설명. summary는 그 내용이 뭘
    보여주는지의 요약.

일반 이미지 프롬프트/응답 파싱(_PROMPT, split_content_summary, extract_mermaid,
캡션 호출의 하드 타임아웃)은 loaders/vlm_caption.py로 뺐다 — image.py(파일
자체가 이미지인 입력)도 크롭 없이 이 로직 그대로 재사용한다. 표 전용 프롬프트와
페이지 크롭(caption_figures)은 PDF에서만 쓰여서 여기 그대로 남아있다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.loaders.vlm_caption import PROMPT as _PROMPT
from document_parser.parsing.loaders.vlm_caption import (
    caption_with_hard_timeout,
    extract_mermaid,
    get_client,
    split_content_summary,
)

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

    from document_parser.parsing.loaders.pdf.layout import LayoutBox

_TABLE_PROMPT = (
    "이 표를 분석해서 아래 형식 그대로, 한국어로만 답해줘(다른 설명 문장 없이 "
    "이 형식만 출력):\n"
    "[CONTENT]\n"
    "표 내용을 마크다운 표 문법으로 그대로 옮겨 적어줘(| 구분자, 헤더 구분줄 "
    "포함).\n"
    "[SUMMARY]\n"
    "표의 내용과 구조를 2~4문장으로 요약(주요 행/열과 특이한 값 언급)."
)

# PP-DocLayoutV2 25개 카테고리 중 수식 관련 3개(layout.py의 ALL_LABELS 참고) —
# 수식은 표/그림과 달리 "내용 자체"가 LaTeX 소스여야 한다는 요청(mermaid와
# 같은 패턴: text 자체가 LaTeX가 됨, metadata["latex"]에도 같은 걸 남김).
_FORMULA_LABELS = {"display_formula", "inline_formula", "formula_number"}

_FORMULA_PROMPT = (
    "이 수식 이미지를 분석해서 아래 형식 그대로, 한국어로만 답해줘(다른 설명 "
    "문장 없이 이 형식만 출력):\n"
    "[CONTENT]\n"
    "수식을 LaTeX 문법으로 그대로 옮겨 적어줘 — 수식 본문만, $$...$$나 "
    "\\[...\\] 같은 구분자나 다른 텍스트 없이 LaTeX 코드만.\n"
    "[SUMMARY]\n"
    "이 수식이 무엇을 나타내는지 1~2문장으로 요약."
)


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    boxes: list[LayoutBox],
) -> list[DocumentElement]:
    if not boxes:
        return []

    client = get_client()
    elements: list[DocumentElement] = []
    for box in boxes:
        pix = page.get_pixmap(clip=box.bbox, dpi=200)
        image_bytes = pix.tobytes("png")
        x0, y0, x1, y1 = box.bbox
        metadata = {
            "source": "vlm",
            # block_type: PP-DocLayoutV2의 25개 카테고리 중 실제 감지된 라벨
            # (예: "chart", "image", "table" 등) — 그림/표라는 것만 아는 게
            # 아니라 어떤 종류인지까지 남겨둔다.
            "block_type": box.label,
            "layout_cls_id": box.cls_id,
            "layout_box_index": box.box_index,
        }

        if box.label == "table":
            element_type = ElementType.TABLE
            result = caption_with_hard_timeout(client, image_bytes, _TABLE_PROMPT)
            text, summary = split_content_summary(result.text)
        elif box.label in _FORMULA_LABELS:
            element_type = ElementType.IMAGE
            result = caption_with_hard_timeout(client, image_bytes, _FORMULA_PROMPT)
            text, summary = split_content_summary(result.text)
            metadata["latex"] = text
        else:
            element_type = ElementType.IMAGE
            result = caption_with_hard_timeout(client, image_bytes, _PROMPT)
            content, summary = split_content_summary(result.text)
            text, mermaid = extract_mermaid(content)
            if mermaid is not None:
                metadata["mermaid"] = mermaid

        if result.usage is not None:
            metadata["vlm_usage"] = result.usage

        elements.append(
            DocumentElement(
                type=element_type,
                text=text,
                summary=summary,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata=metadata,
            )
        )
    return elements
