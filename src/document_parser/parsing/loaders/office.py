"""docx/pptx/ppt/doc 로더 — LibreOffice로 PDF 변환 후 pdf.load() 재사용.

LibreOffice(soffice)는 pip 패키지가 아니라 시스템에 별도 설치해야 하는
바이너리다(예: ``brew install --cask libreoffice``, 리눅스는
``apt install libreoffice``) — paddlepaddle을 전용 인덱스에서 따로 받아야
했던 것과 같은 종류의 외부 의존성. PATH에 soffice/libreoffice가 없으면
MissingDependencyError를 던진다.

변환 후에는 이미 만들어둔 pdf.load()(레이아웃 분석 → native/AzureDI/VLM
라우팅 → 병합)를 그대로 재사용한다 — office 포맷만을 위한 별도 파싱 경로를
새로 만들 필요가 없다.

슬라이드 노트/문서 댓글: 실측으로 확인한 바, LibreOffice의 기본
``--convert-to pdf``는 pptx 슬라이드 노트도 docx 댓글도 export하지
않는다(PDF 변환 시점에 이미 유실됨) — 그래서 PDF 변환과 별도로 원본
zip(Office Open XML은 zip+XML이라 그냥 파일로 열림)에서 직접 뽑는다.
python-pptx/python-docx 같은 새 의존성은 추가하지 않는다(test_office_loader.py
의 기존 방침과 동일 — zipfile + xml.etree만으로 충분).
  - pptx 노트: ``ppt/presentation.xml``의 슬라이드 순서(sldIdLst) ->
    ``presentation.xml.rels``로 실제 slideN.xml 파일 -> 그 슬라이드의
    ``_rels``로 연결된 notesSlideM.xml, 이렇게 관계를 따라가야 한다 —
    슬라이드 번호와 노트 파일 번호가 항상 같지 않다(실제 테스트 파일에서도
    slide7.xml -> notesSlide6.xml처럼 어긋남을 확인함), 그래서 파일명 숫자로
    때려맞추면 틀린다.
  - docx 댓글: ``word/comments.xml``(댓글 본문) + ``word/document.xml``의
    ``commentRangeStart``/``commentRangeEnd`` 사이 텍스트(앵커, 댓글이 달린
    원문)를 뽑는다. 앵커가 어느 페이지/bbox에 해당하는지는 PDF 변환 후에는
    알 수 없으므로(댓글 자체가 변환 과정에서 사라짐), 이미 파싱된 elements
    중 텍스트가 앵커와 가장 많이 겹치는 것을 찾아 그 위치(page/bboxes)를
    빌려 쓴다 — 못 찾으면 위치 없이 텍스트만 남긴다.
  - docx 댓글이 이미지에 달린 경우: PP-DocLayoutV2가 페이지 전체를 block
    단위로 다시 나눠버려서, 원본 텍스트처럼 "내용"으로 대조할 방법이 없다
    (이미지 element의 text는 VLM이 만든 캡션이라 원본 문서 텍스트와 무관) —
    남는 유일한 축은 위치뿐이다. 그래서 댓글 범위 안에 ``w:drawing``
    (``a:blip r:embed``)이 있으면 그 rId로 원본 이미지 바이트를 꺼내고,
    변환된 PDF의 페이지들을 훑어(``page.get_images``) 픽셀 유사도(average
    hash — Pillow 없이 pymupdf만으로 계산, LibreOffice의 재인코딩을
    허용범위로 흡수)로 같은 이미지를 찾은 뒤, 그 bbox와 겹치는(IoU) IMAGE
    element를 찾아 위치를 빌려 쓴다.
"""

from __future__ import annotations

import io
import posixpath
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from document_parser.core.exceptions import MissingDependencyError, ParsingFailedError
from document_parser.core.models import DocumentElement, ElementType
from document_parser.parsing.loaders import pdf

FORMATS = ("docx", "pptx", "ppt", "doc")

_CONVERT_TIMEOUT_SEC = 120

_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _soffice_path() -> str:
    path = shutil.which("soffice") or shutil.which("libreoffice")
    if not path:
        raise MissingDependencyError(
            "docx/pptx/ppt/doc support requires LibreOffice (soffice) installed on PATH "
            "-- e.g. 'brew install --cask libreoffice'"
        )
    return path


def _convert(data: bytes, suffix: str, target_format: str) -> bytes:
    """soffice로 바이트를 다른 포맷으로 변환. PDF 변환(load())과 ppt->pptx
    사전 변환(_extract_pptx_notes) 둘 다 이 헬퍼를 쓴다."""
    soffice = _soffice_path()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(data)

        result = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                target_format,
                "--outdir",
                tmpdir,
                str(input_path),
            ],
            capture_output=True,
            timeout=_CONVERT_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise ParsingFailedError(
                f"LibreOffice conversion to {target_format} failed: "
                f"{result.stderr.decode(errors='replace')}"
            )

        out_path = input_path.with_suffix(f".{target_format}")
        if not out_path.exists():
            raise ParsingFailedError(f"LibreOffice did not produce a .{target_format} output")

        return out_path.read_bytes()


def _extract_pptx_notes(pptx_bytes: bytes) -> list[DocumentElement]:
    with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as zf:
        names = set(zf.namelist())
        if "ppt/presentation.xml" not in names or "ppt/_rels/presentation.xml.rels" not in names:
            return []

        rid_to_target = {
            rel.get("Id"): rel.get("Target")
            for rel in ET.fromstring(zf.read("ppt/_rels/presentation.xml.rels")).iter(
                f"{_REL_NS}Relationship"
            )
        }
        slide_order_rids = [
            sld.get(f"{_R_NS}id")
            for sld in ET.fromstring(zf.read("ppt/presentation.xml")).iter(f"{_P_NS}sldId")
        ]

        elements: list[DocumentElement] = []
        for page_number, rid in enumerate(slide_order_rids, start=1):
            slide_target = rid_to_target.get(rid)
            if not slide_target:
                continue
            slide_path = posixpath.normpath(posixpath.join("ppt", slide_target))
            rels_path = posixpath.join(
                posixpath.dirname(slide_path), "_rels", posixpath.basename(slide_path) + ".rels"
            )
            if rels_path not in names:
                continue

            notes_target = next(
                (
                    rel.get("Target")
                    for rel in ET.fromstring(zf.read(rels_path)).iter(f"{_REL_NS}Relationship")
                    if rel.get("Type", "").endswith("/notesSlide")
                ),
                None,
            )
            if not notes_target:
                continue
            notes_path = posixpath.normpath(
                posixpath.join(posixpath.dirname(slide_path), notes_target)
            )
            if notes_path not in names:
                continue

            text = "".join(
                t.text or "" for t in ET.fromstring(zf.read(notes_path)).iter(f"{_A_NS}t")
            ).strip()
            if not text:
                continue
            elements.append(
                DocumentElement(
                    type=ElementType.NOTE,
                    text=text,
                    page=page_number,
                    metadata={"source": "pptx_notes"},
                )
            )
        return elements


def _docx_comment_spans(document_xml: bytes) -> dict[str, dict[str, object]]:
    """commentRangeStart/End 사이(문서 순서상 임의 개수의 문단/드로잉에 걸칠
    수 있음)에서 텍스트와, 그 안에 들어있는 이미지(``w:drawing``의
    ``a:blip r:embed``) 관계 id를 댓글 id별로 모은다. 동시에 여러 댓글
    범위가 열려 있을 수 있어(중첩) open_spans에 id별로 누적한다."""
    open_spans: dict[str, dict[str, list[str]]] = {}
    spans: dict[str, dict[str, object]] = {}
    for el in ET.fromstring(document_xml).iter():
        tag = el.tag
        if tag == f"{_W_NS}commentRangeStart":
            cid = el.get(f"{_W_NS}id")
            if cid is not None:
                open_spans[cid] = {"text": [], "image_rids": []}
        elif tag == f"{_W_NS}commentRangeEnd":
            cid = el.get(f"{_W_NS}id")
            if cid in open_spans:
                span = open_spans.pop(cid)
                spans[cid] = {
                    "text": "".join(span["text"]).strip(),
                    "image_rids": span["image_rids"],
                }
        elif tag == f"{_W_NS}t":
            for span in open_spans.values():
                span["text"].append(el.text or "")
        elif tag == f"{_A_NS}blip":
            rid = el.get(f"{_R_NS}embed")
            if rid:
                for span in open_spans.values():
                    span["image_rids"].append(rid)
    return spans


def _extract_docx_comments_raw(docx_bytes: bytes) -> list[dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        names = zf.namelist()
        if "word/comments.xml" not in names or "word/document.xml" not in names:
            return []
        comments_xml = zf.read("word/comments.xml")
        document_xml = zf.read("word/document.xml")

    comment_texts: dict[str, str] = {}
    for comment_el in ET.fromstring(comments_xml).iter(f"{_W_NS}comment"):
        cid = comment_el.get(f"{_W_NS}id")
        text = "".join(t.text or "" for t in comment_el.iter(f"{_W_NS}t")).strip()
        if cid is not None and text:
            comment_texts[cid] = text

    spans = _docx_comment_spans(document_xml)
    return [
        {
            "text": text,
            "anchor_text": spans.get(cid, {}).get("text", ""),
            "image_rids": spans.get(cid, {}).get("image_rids", []),
        }
        for cid, text in comment_texts.items()
    ]


# 두 텍스트가 겹치는 정도(0~1) — 앵커가 여러 문단에 걸치는 게 흔해서(실측:
# 댓글 하나가 소제목+불릿 4개 전체에 걸린 경우도 있었음), block 텍스트
# 전체가 앵커 안에 그대로 들어있으면 만점, 아니면 단어 단위 교집합 비율로
# 대체한다 — LibreOffice PDF 변환 과정에서 줄바꿈/공백이 미묘하게 달라져
# 정확한 부분 문자열 매칭이 깨질 수 있어서.
def _match_score(block_text: str, anchor_text: str) -> float:
    block_text = block_text.strip()
    if not block_text or not anchor_text:
        return 0.0
    if block_text in anchor_text:
        return 1.0
    block_tokens = set(block_text.split())
    if not block_tokens:
        return 0.0
    anchor_tokens = set(anchor_text.split())
    return len(block_tokens & anchor_tokens) / len(block_tokens)


# 이 밑으로는 "겹친다"고 보지 않는다 — 우연히 흔한 단어 몇 개만 겹치는
# 무관한 block까지 매칭시키지 않기 위한 최소 기준.
_COMMENT_MATCH_MIN_SCORE = 0.15


def _best_matching_element(
    anchor_text: str, elements: list[DocumentElement]
) -> DocumentElement | None:
    best: DocumentElement | None = None
    best_score = _COMMENT_MATCH_MIN_SCORE
    for el in elements:
        if el.type == ElementType.NOTE or not el.text:
            continue
        score = _match_score(el.text, anchor_text)
        if score > best_score:
            best_score = score
            best = el
    return best


def _resolve_docx_media(zf: zipfile.ZipFile, rid: str) -> bytes | None:
    """word/_rels/document.xml.rels에서 rId로 media 파일(원본 이미지)을 찾는다."""
    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return None
    target = next(
        (
            rel.get("Target")
            for rel in ET.fromstring(zf.read(rels_path)).iter(f"{_REL_NS}Relationship")
            if rel.get("Id") == rid
        ),
        None,
    )
    if not target:
        return None
    media_path = posixpath.normpath(posixpath.join("word", target))
    if media_path not in zf.namelist():
        return None
    return zf.read(media_path)


# 8x8 average hash(64비트)만으로 충분 — 문서에 박힌 그림 몇 장 중에서
# "이게 그거다"만 가려내면 되고, 지문 검색처럼 대규모 데이터베이스에서
# 찾는 게 아니라서 더 정교한 해시(pHash/dHash)까진 필요 없다.
_IMAGE_HASH_SIZE = 8
# 64비트 중 이 이하 차이는 "같은 이미지"로 본다 — LibreOffice가 PDF로 구우며
# 원본을 재인코딩(포맷 변경 등)해도 픽셀 근사값은 유지되는 걸 실측으로
# 확인함(재인코딩 후 거리 0, 실제로 다른 그림끼리는 20+ 나옴).
_IMAGE_HASH_MAX_DISTANCE = 10


def _perceptual_hash(image_bytes: bytes, size: int = _IMAGE_HASH_SIZE) -> int | None:
    """average hash(aHash) — Pillow를 새 의존성으로 추가하지 않고 pymupdf
    Pixmap만으로 계산한다(office.py의 기존 방침: 새 의존성 추가 안 함)."""
    import pymupdf

    try:
        pix = pymupdf.Pixmap(image_bytes)
    except Exception:
        return None
    if pix.colorspace is not None and pix.colorspace.n > 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    small = pymupdf.Pixmap(pix, size, size)
    samples = small.samples
    n = small.n
    values = [samples[i * n] for i in range(size * size)]
    avg = sum(values) / len(values)
    bits = 0
    for v in values:
        bits = (bits << 1) | (1 if v >= avg else 0)
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _find_image_location_in_pdf(
    pdf_bytes: bytes, image_bytes: bytes
) -> tuple[int, tuple[float, float, float, float]] | None:
    """원본 docx에 박힌 이미지가 변환된 PDF의 어느 페이지/bbox에 해당하는지
    찾는다. rId 같은 안정적인 식별자는 변환 과정에서 사라지므로(LibreOffice가
    PDF로 구울 때 원본 관계 정보를 남기지 않음), 픽셀 유사도로 대조하는
    수밖에 없다."""
    target_hash = _perceptual_hash(image_bytes)
    if target_hash is None:
        return None

    import pymupdf

    best: tuple[int, tuple[float, float, float, float]] | None = None
    best_distance = _IMAGE_HASH_MAX_DISTANCE + 1
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    extracted = doc.extract_image(xref)
                except Exception:
                    continue
                candidate_hash = _perceptual_hash(extracted["image"])
                if candidate_hash is None:
                    continue
                distance = _hamming_distance(target_hash, candidate_hash)
                if distance < best_distance:
                    bbox = page.get_image_bbox(img)
                    best = (page_number, (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
                    best_distance = distance
    return best if best_distance <= _IMAGE_HASH_MAX_DISTANCE else None


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


# 레이아웃 모델이 잡은 박스가 원본 이미지 그대로의 extent와 약간 다를 수
# 있어서(크롭 여백 등) 느슨하게 잡는다 — 표/텍스트 매칭의 0.15(단어 겹침
# 비율)와 스케일이 달라 별도 상수로 둔다.
_IMAGE_BBOX_MIN_IOU = 0.3


def _best_matching_bbox(
    page_number: int,
    bbox: tuple[float, float, float, float],
    elements: list[DocumentElement],
) -> DocumentElement | None:
    best: DocumentElement | None = None
    best_iou = _IMAGE_BBOX_MIN_IOU
    for el in elements:
        if el.type != ElementType.IMAGE or el.page != page_number:
            continue
        for el_bbox in el.bboxes:
            iou = _bbox_iou(bbox, (el_bbox.x0, el_bbox.y0, el_bbox.x1, el_bbox.y1))
            if iou > best_iou:
                best_iou = iou
                best = el
    return best


def _match_comment_image(
    docx_bytes: bytes,
    pdf_bytes: bytes,
    image_rids: list[str],
    elements: list[DocumentElement],
) -> DocumentElement | None:
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        for rid in image_rids:
            media_bytes = _resolve_docx_media(zf, rid)
            if media_bytes is None:
                continue
            location = _find_image_location_in_pdf(pdf_bytes, media_bytes)
            if location is None:
                continue
            matched = _best_matching_bbox(*location, elements)
            if matched is not None:
                return matched
    return None


def _extract_docx_comments(
    docx_bytes: bytes,
    elements: list[DocumentElement],
    pdf_bytes: bytes | None = None,
) -> list[DocumentElement]:
    result: list[DocumentElement] = []
    for comment in _extract_docx_comments_raw(docx_bytes):
        anchor = comment["anchor_text"]
        image_rids = comment["image_rids"]

        matched = None
        if image_rids and pdf_bytes is not None:
            matched = _match_comment_image(docx_bytes, pdf_bytes, image_rids, elements)
        if matched is None and anchor:
            matched = _best_matching_element(anchor, elements)

        result.append(
            DocumentElement(
                type=ElementType.NOTE,
                text=comment["text"],
                page=matched.page if matched else None,
                bboxes=matched.bboxes if matched else [],
                metadata={"source": "docx_comment", "anchor_text": anchor},
            )
        )
    return result


def load(data: bytes, source: str, tier: str = "balanced") -> list[DocumentElement]:
    suffix = Path(source).suffix or ".docx"
    fmt = suffix.lstrip(".").lower()

    pdf_bytes = _convert(data, suffix, "pdf")
    elements = pdf.load(pdf_bytes, source, tier)

    if fmt == "pptx":
        elements += _extract_pptx_notes(data)
    elif fmt == "ppt":
        elements += _extract_pptx_notes(_convert(data, suffix, "pptx"))
    elif fmt == "docx":
        elements += _extract_docx_comments(data, elements, pdf_bytes)

    return elements
