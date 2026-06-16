"""Document ingestion: parse files (type-aware) and chunk by Q&A structure."""
import re
from pathlib import Path

import fitz  # PyMuPDF — PDFs only


def extract_file_text(file_path: str) -> list[dict]:
    """Route by file type. Fail loudly on anything unexpected."""
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name

    if ext == ".pdf":
        doc = fitz.open(file_path)
        return [
            {"page_number": i + 1, "text": page.get_text(), "source": name}
            for i, page in enumerate(doc)
        ]
    elif ext in (".txt", ".md"):
        # a .txt has no real "pages" — be honest about that
        return [{
            "page_number": None,
            "text": Path(file_path).read_text(encoding="utf-8"),
            "source": name,
        }]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_by_qa(page_text: str, source: str, page_number) -> list[dict]:
    """Split FAQ-style text into one chunk per 'Q<x>.<y>:' Q&A pair."""
    parts = re.split(r"(?=Q\d+\.\d+:)", page_text)
    chunks = []
    for part in parts:
        part = part.strip()
        if len(part) < 20:  # skip headers / empty fragments
            continue
        chunks.append({"text": part, "source": source, "page_number": page_number})
    return chunks


def build_chunks(file_paths: list[str]) -> list[dict]:
    """Full ingestion pipeline: files -> pages -> Q&A chunks with stable ids."""
    chunks: list[dict] = []
    for fp in file_paths:
        for page in extract_file_text(fp):
            chunks.extend(chunk_by_qa(page["text"], page["source"], page["page_number"]))
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    return chunks
