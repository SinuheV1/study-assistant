import hashlib
import re
from pathlib import Path
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize

from src.utils.logging import setup_logger

log = setup_logger(__name__)

_PUNKT_READY = False


def ensure_nltk_punkt() -> None:
    """
    Ensure NLTK sentence tokenizer resources are available.

    This uses the standard NLTK downloader. No SSL or certificate workaround
    is applied here. If downloading fails, the environment needs to be fixed
    outside this function.
    """
    global _PUNKT_READY

    if _PUNKT_READY:
        return

    nltk_data_dir = Path.home() / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)

    if str(nltk_data_dir) not in nltk.data.path:
        nltk.data.path.append(str(nltk_data_dir))

    required_resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]

    for package_name, resource_path in required_resources:
        try:
            nltk.data.find(resource_path)
            log.info(f"NLTK resource already available: {package_name}")
            continue

        except LookupError:
            log.info(f"Downloading NLTK resource: {package_name}")

            downloaded = nltk.download(
                package_name,
                download_dir=str(nltk_data_dir),
                quiet=False,
                raise_on_error=False,
            )

            if not downloaded:
                raise RuntimeError(
                    f"Failed to download NLTK resource '{package_name}'. "
                    "Try running manually: "
                    f"python -m nltk.downloader -d {nltk_data_dir} {package_name}"
                )

        # Verify after download.
        try:
            nltk.data.find(resource_path)
        except LookupError as exc:
            raise RuntimeError(
                f"NLTK resource '{package_name}' was downloaded but could not be found. "
                f"Expected resource path: {resource_path}. "
                f"Download directory: {nltk_data_dir}"
            ) from exc

    _PUNKT_READY = True


def split_block_into_sentences(block: str) -> list[str]:
    """
    Split a paragraph/block into sentence-safe units using NLTK.

    Purpose:
    - Avoid cutting textbook chunks mid-sentence.
    - Keep chunk text more readable.
    - Make overlap sentence-based instead of character-based.
    """
    if not block:
        return []

    ensure_nltk_punkt()

    cleaned_block = " ".join(block.split())

    if not cleaned_block:
        return []

    sentences = sent_tokenize(cleaned_block)

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_page_into_blocks(page_text: str) -> list[str]:
    """
    Split one page of textbook Markdown/text into logical blocks.

    Blocks usually represent:
    - headings
    - paragraphs
    - lists
    - equations
    - captions

    This function does not split into sentences yet.
    """
    if not page_text:
        return []

    text = page_text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line, but preserve blank lines as paragraph boundaries.
    lines = [line.strip() for line in text.split("\n")]
    normalized_text = "\n".join(lines)

    # Split on blank lines.
    raw_blocks = re.split(r"\n\s*\n+", normalized_text)

    blocks = []

    for block in raw_blocks:
        cleaned_block = " ".join(block.split())

        if not cleaned_block:
            continue

        blocks.append(cleaned_block)

    return blocks


def extract_page_blocks(
    cleaned_text: str,
    document_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract page-level records from cleaned textbook text.

    Output:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."}
    ]

    Page blocks are used for citation metadata.
    They should not force chunk boundaries.
    """
    if not cleaned_text:
        return []

    # Best case: ingestion already saved structured page data.
    pages = document_metadata.get("pages")

    if isinstance(pages, list) and pages:
        page_blocks = []

        for index, page in enumerate(pages, start=1):
            if isinstance(page, dict):
                page_number = page.get("page") or page.get("page_number") or index
                page_text = page.get("text", "")
            else:
                page_number = index
                page_text = str(page)

            if page_text and page_text.strip():
                page_blocks.append(
                    {
                        "page": page_number,
                        "text": page_text.strip(),
                    }
                )

        if page_blocks:
            return page_blocks

    text = cleaned_text.strip()

    # Some PDF extractors preserve page breaks with form feed.
    if "\f" in text:
        page_texts = [page.strip() for page in text.split("\f") if page.strip()]

        return [
            {
                "page": index,
                "text": page_text,
            }
            for index, page_text in enumerate(page_texts, start=1)
        ]

    # Optional explicit page markers.
    # Supported:
    # <PAGE 1>
    # --- Page 1 ---
    # <!-- page: 1 -->
    page_marker_pattern = re.compile(
        r"(?:^|\n)\s*(?:"
        r"<PAGE\s+(\d+)>"
        r"|---\s*Page\s+(\d+)\s*---"
        r"|<!--\s*page:\s*(\d+)\s*-->"
        r")\s*(?:\n|$)",
        flags=re.IGNORECASE,
    )

    matches = list(page_marker_pattern.finditer(text))

    if matches:
        page_blocks = []

        for i, match in enumerate(matches):
            page_number_text = next(group for group in match.groups() if group is not None)
            page_number = int(page_number_text)

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            page_text = text[start:end].strip()

            if page_text:
                page_blocks.append(
                    {
                        "page": page_number,
                        "text": page_text,
                    }
                )

        if page_blocks:
            return page_blocks

    # Fallback: no page information available.
    return [
        {
            "page": None,
            "text": text,
        }
    ]


def is_textbook_heading(block: str) -> bool:
    """
    Detect likely textbook headings.

    This is intentionally strict to avoid treating glossary terms,
    margin notes, figure labels, or short phrases as section headings.

    Accepts:
    - Markdown headings produced from Docling section_header labels
    - Chapter-style headings
    - Numbered textbook headings like:
        2 Statistical Learning
        2.1 What Is Statistical Learning?
        2.1.4 Supervised Versus Unsupervised Learning

    Rejects:
    - random short phrases like "expected test MSE"
    - glossary/margin terms like "input variable output variable"
    - normal sentences
    """
    if not block:
        return False

    text = " ".join(block.strip().split())

    if not text:
        return False

    # Do not treat long blocks as headings.
    if len(text) > 140:
        return False

    # Markdown headings from Docling provenance exporter:
    # ## Statistical Learning
    # ## 2.1 What Is Statistical Learning?
    if re.match(r"^#{1,6}\s+\S+", text):
        return True

    # Explicit chapter headings:
    # Chapter 2
    # Chapter 2 Statistical Learning
    # Ch. 2 Statistical Learning
    if re.match(r"^(chapter|ch\.)\s+\d+(\s+.+)?$", text, flags=re.IGNORECASE):
        return True

    # Numbered textbook headings:
    # 2 Statistical Learning
    # 2.1 What Is Statistical Learning?
    # 2.1.4 Supervised Versus Unsupervised Learning
    if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9,()\-:;? ]+$", text):
        return True

    return False


def normalize_textbook_heading(block: str) -> str:
    """
    Normalize a textbook heading for metadata.
    """
    heading = block.strip()

    # Remove Markdown heading markers.
    heading = re.sub(r"^#{1,6}\s+", "", heading)

    # Collapse whitespace.
    heading = " ".join(heading.split())

    return heading.strip()


def heading_is_chapter(heading: str) -> bool:
    """
    Determine whether a heading is chapter-level.
    """
    if not heading:
        return False

    return bool(re.match(r"^(chapter|ch\.)\s+\d+", heading, flags=re.IGNORECASE))


def build_textbook_units(page_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert page blocks into sentence-level units with page and section metadata.

    Unit format:
    {
        "text": "...",
        "page": 14,
        "chapter": "Chapter 3",
        "section": "Least Squares"
    }
    """
    units = []

    current_chapter = "Unknown"
    current_section = "Unknown"

    for page_block in page_blocks:
        page_number = page_block.get("page")
        page_text = page_block.get("text", "")

        blocks = split_page_into_blocks(page_text)

        for block in blocks:
            if is_textbook_heading(block):
                heading = normalize_textbook_heading(block)

                if heading_is_chapter(heading):
                    current_chapter = heading
                    current_section = heading

                elif current_chapter == "Unknown":
                    current_chapter = heading
                    current_section = heading

                else:
                    current_section = heading

                continue

            sentences = split_block_into_sentences(block)

            for sentence in sentences:
                units.append(
                    {
                        "text": sentence,
                        "page": page_number,
                        "chapter": current_chapter,
                        "section": current_section,
                    }
                )

    return units


def get_sentence_overlap_units(
    current_units: list[dict[str, Any]],
    overlap_size: int,
) -> list[dict[str, Any]]:
    """
    Create overlap from full sentence units using a character budget.

    This avoids carrying partial words or partial sentences.
    """
    if not current_units or overlap_size <= 0:
        return []

    overlap_units = []
    total_length = 0

    for unit in reversed(current_units):
        unit_length = len(unit.get("text", ""))

        if total_length + unit_length > overlap_size:
            break

        overlap_units.insert(0, unit)
        total_length += unit_length

    return overlap_units


def finalize_textbook_chunk(units: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert accumulated sentence units into one chunk object.
    """
    chunk_text = " ".join(unit.get("text", "") for unit in units).strip()

    known_pages = [unit.get("page") for unit in units if isinstance(unit.get("page"), int)]

    chapters = []
    sections = []

    for unit in units:
        chapter = unit.get("chapter", "Unknown")
        section = unit.get("section", "Unknown")

        if chapter not in chapters:
            chapters.append(chapter)

        if section not in sections:
            sections.append(section)

    page_start = min(known_pages) if known_pages else None
    page_end = max(known_pages) if known_pages else None

    return {
        "text": chunk_text,
        "page_start": page_start,
        "page_end": page_end,
        "chapter": chapters[-1] if chapters else "Unknown",
        "chapters": chapters,
        "section": sections[-1] if sections else "Unknown",
        "sections": sections,
        "unit_count": len(units),
    }


def build_chunks_from_units(
    units: list[dict[str, Any]],
    target_size: int,
    overlap_size: int,
    min_chunk_size: int = 250,
) -> list[dict[str, Any]]:
    """
    Build textbook chunks from sentence units.

    Important behavior:
    - Allows chunks to span pages.
    - Does not split mid-sentence.
    - Prefers not to cross section boundaries.
    - Uses sentence-based overlap.
    """
    if not units:
        return []

    chunks = []

    current_units = []
    current_length = 0
    current_section = None

    for unit in units:
        unit_text = unit.get("text", "")
        unit_length = len(unit_text)
        unit_section = unit.get("section", "Unknown")

        section_changed = current_section is not None and unit_section != current_section

        if section_changed and current_units and current_length >= min_chunk_size:
            chunks.append(finalize_textbook_chunk(current_units))

            overlap_units = get_sentence_overlap_units(
                current_units=current_units,
                overlap_size=overlap_size,
            )

            current_units = overlap_units
            current_length = sum(len(u.get("text", "")) for u in current_units)

        candidate_length = current_length + unit_length

        if candidate_length <= target_size or not current_units:
            current_units.append(unit)
            current_length += unit_length
            current_section = unit_section
            continue

        chunks.append(finalize_textbook_chunk(current_units))

        overlap_units = get_sentence_overlap_units(
            current_units=current_units,
            overlap_size=overlap_size,
        )

        current_units = overlap_units + [unit]
        current_length = sum(len(u.get("text", "")) for u in current_units)
        current_section = unit_section

    if current_units:
        chunks.append(finalize_textbook_chunk(current_units))

    return chunks


def create_textbook_chunk_id(
    document_id: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """
    Create stable-ish chunk id from document id, index, and text hash.
    """
    raw = f"{document_id}_{chunk_index}_{chunk_text}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    return f"chunk_{digest}"


def create_textbook_chunk_metadata(
    document_metadata: dict[str, Any],
    chunk_object: dict[str, Any],
    chunk_index: int,
    total_chunks: int,
) -> dict[str, Any]:
    """
    Create metadata for one textbook chunk.
    """
    chunk_text = chunk_object.get("text", "")
    document_id = document_metadata.get("document_id", "unknown_document")

    metadata = dict(document_metadata)

    # Avoid copying large page payloads into every chunk metadata.
    metadata.pop("pages", None)

    chunk_id = create_textbook_chunk_id(
        document_id=document_id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
    )

    metadata.update(
        {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "source_type": "textbook_pdf",
            "chapter": chunk_object.get("chapter", "Unknown"),
            "chapters": chunk_object.get("chapters", []),
            "section": chunk_object.get("section", "Unknown"),
            "sections": chunk_object.get("sections", []),
            "page_start": chunk_object.get("page_start"),
            "page_end": chunk_object.get("page_end"),
            "unit_count": chunk_object.get("unit_count", 0),
            "chunk_text_length": len(chunk_text),
        }
    )

    return metadata


def chunk_textbook_pdf(
    cleaned_text: str,
    document_metadata: dict[str, Any],
    target_size: int,
    overlap_size: int,
) -> list[dict[str, Any]]:
    """
    Main textbook PDF chunking strategy.

    Designed for large textbook PDFs where content may span multiple pages.

    Strategy:
    - Preserve page metadata.
    - Split pages into blocks.
    - Split blocks into NLTK sentence units.
    - Build sentence-safe chunks.
    - Allow chunks to span page boundaries.
    - Store page_start/page_end for citations.
    """
    page_blocks = extract_page_blocks(
        cleaned_text=cleaned_text,
        document_metadata=document_metadata,
    )

    if not page_blocks:
        log.warning(
            "No page blocks found for textbook document "
            f"{document_metadata.get('document_id', 'unknown')}."
        )
        return []

    units = build_textbook_units(page_blocks)

    if not units:
        log.warning(
            "No sentence units created for textbook document "
            f"{document_metadata.get('document_id', 'unknown')}."
        )
        return []

    chunk_objects = build_chunks_from_units(
        units=units,
        target_size=target_size,
        overlap_size=overlap_size,
    )

    total_chunks = len(chunk_objects)
    chunk_records = []

    for index, chunk_object in enumerate(chunk_objects):
        chunk_text = chunk_object.get("text", "")

        chunk_metadata = create_textbook_chunk_metadata(
            document_metadata=document_metadata,
            chunk_object=chunk_object,
            chunk_index=index,
            total_chunks=total_chunks,
        )

        chunk_records.append(
            {
                "chunk_id": chunk_metadata["chunk_id"],
                "chunk_text": chunk_text,
                "metadata": chunk_metadata,
            }
        )

    log.info(
        f"Textbook document '{document_metadata.get('document_id', 'unknown')}' "
        f"chunked into {total_chunks} chunks."
    )

    return chunk_records
