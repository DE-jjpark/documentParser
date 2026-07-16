"""docx/pptx/ppt/doc 로더 — LibreOffice로 PDF 변환 후 pdf.load() 재사용.

LibreOffice(soffice)는 pip 패키지가 아니라 시스템에 별도 설치해야 하는
바이너리다(예: ``brew install --cask libreoffice``, 리눅스는
``apt install libreoffice``) — paddlepaddle을 전용 인덱스에서 따로 받아야
했던 것과 같은 종류의 외부 의존성. PATH에 soffice/libreoffice가 없으면
MissingDependencyError를 던진다.

변환 후에는 이미 만들어둔 pdf.load()(레이아웃 분석 → native/AzureDI/VLM
라우팅 → 병합)를 그대로 재사용한다 — office 포맷만을 위한 별도 파싱 경로를
새로 만들 필요가 없다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from document_parser.core.exceptions import MissingDependencyError, ParsingFailedError
from document_parser.core.models import DocumentElement
from document_parser.parsing.loaders import pdf

FORMATS = ("docx", "pptx", "ppt", "doc")

_CONVERT_TIMEOUT_SEC = 120


def _soffice_path() -> str:
    path = shutil.which("soffice") or shutil.which("libreoffice")
    if not path:
        raise MissingDependencyError(
            "docx/pptx/ppt/doc support requires LibreOffice (soffice) installed on PATH "
            "-- e.g. 'brew install --cask libreoffice'"
        )
    return path


def load(data: bytes, source: str) -> list[DocumentElement]:
    soffice = _soffice_path()
    suffix = Path(source).suffix or ".docx"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(data)

        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmpdir, str(input_path)],
            capture_output=True,
            timeout=_CONVERT_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise ParsingFailedError(
                f"LibreOffice conversion failed for {source}: "
                f"{result.stderr.decode(errors='replace')}"
            )

        pdf_path = input_path.with_suffix(".pdf")
        if not pdf_path.exists():
            raise ParsingFailedError(f"LibreOffice did not produce a PDF for {source}")

        pdf_bytes = pdf_path.read_bytes()

    return pdf.load(pdf_bytes, source)
