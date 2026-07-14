"""200dpi 렌더링 픽셀 좌표 <-> pymupdf 포인트(72dpi) 좌표 변환.

PP-DocLayoutV2도 Azure Document Intelligence(이미지 입력)도 페이지를
RENDER_DPI로 렌더링한 이미지 기준 픽셀 좌표로 bbox를 준다. pymupdf(clip 등)는
포인트 좌표를 쓰므로, 이 변환을 놓치면 실제 위치보다 RENDER_DPI/72배(200dpi
기준 약 2.78배) 어긋난 엉뚱한 영역을 가리키게 된다 — 실제로 이 버그로 텍스트
전용 페이지에서 clip 영역이 빗나가 요소가 0개 나온 적이 있어(layout.py 관련
커밋 참고) 공용 유틸로 빼서 레이아웃 분석/AzureDI/VLM이 전부 같은 변환을
쓰게 한다.
"""

from __future__ import annotations

RENDER_DPI = 200
_PX_TO_PT = 72 / RENDER_DPI


def px_to_pt(coordinate: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(v * _PX_TO_PT for v in coordinate)
