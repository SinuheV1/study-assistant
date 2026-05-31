from docling.document_converter import DocumentConverter
from src.utils.logging import setup_logger
from src.ingestion.clean_text import basic_text_cleaning
from src.ingestion.ingest_text import build_document_metadata
import os
log = setup_logger(__name__)

def export_docling_to_page_marked_markdown(doc):
    """
    Export a DoclingDocument to page-marked Markdown using Docling item provenance.

    This avoids page-by-page PDF reconversion and preserves page numbers from
    Docling's internal document structure.

    Output format:
    <PAGE 1>

    text...

    <PAGE 2>

    text...
    """
    if not doc:
        log.warning("Docling document does not exist or is empty.")
        return ""

    page_blocks = {}

    try:
        for item, level in doc.iterate_items():
            prov = getattr(item, "prov", None)

            if not prov:
                continue

            page_number = getattr(prov[0], "page_no", None)

            if page_number is None:
                continue

            text = getattr(item, "text", None)

            if not text:
                # Some Docling items may not expose plain text.
                # Skip them for now rather than breaking ingestion.
                continue

            text = text.strip()

            if not text:
                continue

            label = getattr(item, "label", None)
            label_value = getattr(label, "value", str(label)).lower() if label else ""

            if label_value in ["title", "section_header"]:
                line = f"\n## {text}\n"
            elif label_value == "list_item":
                line = f"- {text}"
            else:
                line = text

            page_blocks.setdefault(page_number, []).append(line)

        if not page_blocks:
            log.warning("No provenance-backed page blocks found in Docling document.")
            return ""

        markdown_pages = []

        for page_number in sorted(page_blocks):
            page_text = "\n\n".join(page_blocks[page_number]).strip()

            if not page_text:
                continue

            markdown_pages.append(f"<PAGE {page_number}>\n\n{page_text}")

        markdown_text = "\n\n".join(markdown_pages)

        log.info(
            f"Docling document exported with provenance page markers. "
            f"Pages found: {len(markdown_pages)}"
        )

        return markdown_text

    except Exception as e:
        log.warning(f"Error exporting Docling document with provenance: {e}")
        return ""
    
    
def convert_document_with_docling(file_path):
    if not os.path.exists(file_path):
        log.warning(f'Path does not exist. {file_path}')
        return None
    if not file_path:
        log.warning(f'File Path does not exist : {file_path}. Verify path exists and try again.')
        return None
    try:
        converter=DocumentConverter()
        doc=converter.convert(file_path).document
        if not doc:
            log.warning(f'Document conversion for {file_path} failed.')
            return None
        log.info(f'PDF successfully converted : {file_path}')
        return doc
    except Exception as e:
        log.warning(f'Error converting document. {e}')
        return None
    
def export_docling_to_markdown(conversion_result):
    if not conversion_result:
        log.warning(f'Conversion result document does not exist or is empty.')
        return None
    try:
        markdown_text=conversion_result.export_to_markdown()
        if not markdown_text:
            log.warning(f'Document not converted to markdown.')
            return ''
        log.info(f'Document successfuly converted to markdown.')
        return markdown_text
    except Exception as e:
        log.warning(f'Error occured during conversion. {e}')
        return ''
    
def assess_extraction_quality(extracted_text,file_path):
    if extracted_text is None:
        log.warning(f'Extraction failed for {file_path}. Extracted text is None.')
        return False
    text=extracted_text.strip()
    if not text:
        log.warning(f'Extraction failed for {file_path}. Extracted text is empty.')
        return False
    char_count=len(text)
    if char_count < 200:
        log.warning(f'Extracted text quality failed for {file_path}. Returned {char_count} characters. ')
        return False
    count_alphabetic= sum(char.isalpha() for char in text)
    alphabetic_ratio=count_alphabetic/char_count
    if alphabetic_ratio< 0.45:
        log.warning(f'Extracted text quality failed for {file_path}. Low alphabetic ratio {alphabetic_ratio:.2f} ')
        return False
    count_whitespace=sum(char.isspace() for char in text)
    whitespace_ratio=count_whitespace/char_count
    if whitespace_ratio>0.55:
        log.warning(f'Extracted text quality failed for {file_path}. High whitespace ratio {whitespace_ratio:.2f} ')
    weird_symbol_count=sum(1 for char in text if not char.isalnum() and not char.isspace() 
                        and char not in ".,;:!?()[]{}+-=*/%$#@&'\"“”‘’<>|\\")
    weird_symbol_ratio=weird_symbol_count/char_count
    if weird_symbol_ratio>0.15:
        log.warning(f'Extracted text quality failed for {file_path}. High weird symbol ratio {weird_symbol_ratio:.2f} ')
        return False
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <3:
        log.warning(f'Extraction failed for document {file_path}. Too few non empty lines.')
        return False
    avg_line_len=sum(len(line) for line in lines)/len(lines)
    if avg_line_len <15:
        log.warning(f'Extraction failed for document {file_path}. Very short average line length.')
        #return False
    log.info(f'Extraction passed for {file_path}:'
            f'{char_count} characters, alphabetic ratio = {alphabetic_ratio:.2f}, '
            f'weird symbol ratio = {weird_symbol_ratio:.2f}')
    return True
    
def ingest_docling_document(file_path):
    log.info("Starting docling document ingestion.")

    metadata = build_document_metadata(file_path)
    source_type = metadata.get("source_type")

    doc = convert_document_with_docling(file_path)

    if doc is None:
        log.critical("File unsuccessfully converted. Docling document returned None.")
        return None

    if source_type == "textbook_pdf":
        log.info("Detected textbook PDF. Using Docling provenance-aware page export.")
        raw_text = export_docling_to_page_marked_markdown(doc)
        metadata["page_marker_strategy"] = "docling_item_provenance"
    else:
        raw_text = export_docling_to_markdown(doc)

    if not raw_text:
        log.critical("Document unsuccessfully converted to markdown.")
        return None

    is_valid = assess_extraction_quality(raw_text, file_path)

    if not is_valid:
        log.warning(f"Skipping document due to poor extraction results. {file_path}")
        return None

    cleaned_text = basic_text_cleaning(raw_text)

    metadata["extraction_method"] = "docling"
    metadata["raw_extraction_format"] = "markdown_with_page_markers"

    log.info("Document successfully extracted with docling.")

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "metadata": metadata,
    }