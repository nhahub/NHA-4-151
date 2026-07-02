"""
PDF Parser — Extract text from uploaded PDF resume files.

Primary: PyMuPDF (fitz) — fast and reliable.
Fallback: pdfplumber — handles complex layouts.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    """
    Extract text from PDF bytes using multiple strategies.

    Args:
        pdf_bytes: Raw PDF file content as bytes.

    Returns:
        Extracted text string, or None if extraction fails.
    """
    text = _try_pymupdf(pdf_bytes)

    if not text or len(text.strip()) < 50:
        logger.warning("PyMuPDF extraction insufficient. Trying pdfplumber.")
        text = _try_pdfplumber(pdf_bytes)

    if text:
        text = _clean_text(text)

    return text


def _try_pymupdf(pdf_bytes: bytes) -> Optional[str]:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text.strip():
                pages.append(page_text)
        result = "\n\n".join(pages)
        logger.info(f"PyMuPDF extracted {len(result)} chars from {doc.page_count} pages.")
        doc.close()
        return result

    except ImportError:
        logger.warning("PyMuPDF not installed. Skipping.")
        return None
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return None


def _try_pdfplumber(pdf_bytes: bytes) -> Optional[str]:
    """Extract text using pdfplumber."""
    try:
        import pdfplumber
        import io

        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append(page_text)

        result = "\n\n".join(pages)
        logger.info(f"pdfplumber extracted {len(result)} chars from {len(pages)} pages.")
        return result

    except ImportError:
        logger.warning("pdfplumber not installed. Skipping.")
        return None
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return None


def _clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    - Remove excessive whitespace
    - Normalize line breaks
    - Remove common PDF artifacts
    """
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3+ newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove null bytes and other control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
