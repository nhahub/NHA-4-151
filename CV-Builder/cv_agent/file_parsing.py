"""
cv_agent.file_parsing — Resume file parsing (PDF, DOCX, TXT).
"""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Union
from cv_agent.config import _PDFPLUMBER_AVAILABLE

def parse_resume_bytes(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if _PDFPLUMBER_AVAILABLE:
            import pdfplumber
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        else:
            try:
                from pypdf import PdfReader
                return "\n".join(pg.extract_text() or "" for pg in PdfReader(BytesIO(file_bytes)).pages)
            except ImportError as e:
                raise ImportError("PDF parsing requires pdfplumber or pypdf") from e
    elif ext in (".docx", ".doc"):
        try:
            import docx
            return "\n".join(para.text for para in docx.Document(BytesIO(file_bytes)).paragraphs)
        except ImportError as e:
            raise ImportError("DOCX parsing requires python-docx") from e
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt")

def parse_resume_file(path: Union[str, Path]) -> str:
    path = Path(path)
    return parse_resume_bytes(path.read_bytes(), path.name)
