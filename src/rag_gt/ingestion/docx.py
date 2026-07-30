"""DOCX text extraction. Opt-in via config `ingestion.enable_docx: true`.

PDFs are the primary supported format; DOCX is kept for future/optional use.

Headings are emitted as their own line prefixed with `# ` so the heading-aware
chunker (`chunking.strategies._heading_semantic`) can detect them.
"""

from __future__ import annotations


def extract_docx(path: str) -> str:
    try:
        from docx import Document as DocxDoc  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX ingestion. "
            "Install with: pip install python-docx"
        ) from e

    try:
        doc = DocxDoc(path)
    except Exception as e:
        raise ValueError(f"Failed to open DOCX file {path!r}: {e}") from e

    sections: list[list[str]] = []
    current: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        style = (para.style.name or "").lower() if para.style else ""
        if not text:
            if current:
                sections.append(current)
                current = []
            continue
        if "heading" in style:
            if current:
                sections.append(current)
                current = []
            current.append(f"# {text}")
            continue
        current.append(text)
    if current:
        sections.append(current)

    # Skip sections that contain only a heading line with no body — they create
    # empty "chunks" downstream. Keep their heading by merging into the next
    # section if one exists.
    merged: list[list[str]] = []
    pending_heading: list[str] = []
    for section in sections:
        body_lines = [ln for ln in section if not ln.startswith("# ")]
        if not body_lines:
            pending_heading.extend(section)
            continue
        if pending_heading:
            section = pending_heading + section
            pending_heading = []
        merged.append(section)
    if pending_heading and merged:
        merged[-1] = pending_heading + merged[-1]
    elif pending_heading:
        merged.append(pending_heading)

    # Within a section, join lines with newlines so heading boundaries survive.
    return "\n\n".join("\n".join(section) for section in merged)
