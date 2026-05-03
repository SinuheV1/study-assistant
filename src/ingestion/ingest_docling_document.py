from docling.document_converter import DocumentConverter
from src.utils.logging import setup_logger
from src.ingestion.clean_text import basic_text_cleaning
from src.ingestion.ingest_text import build_document_metadata
import os
log = setup_logger(__name__)

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
    log.info(f'Starting docling document ingestion. ')
    conversion_result=convert_document_with_docling(file_path)
    if conversion_result is None:
        log.critical(f'File unsuccessfully converted. Conversion Result returned None.')
        return None
    raw_text=export_docling_to_markdown(conversion_result)
    if not raw_text:
        log.critical(f'Document unsuccessfully converted to markdown. ')
        return None
    is_valid=assess_extraction_quality(raw_text,file_path)
    if not is_valid:
        log.warning(f'Skipping document due to poor extraction results. {file_path} ')
        return None
    cleaned_text=basic_text_cleaning(raw_text)
    metadata=build_document_metadata(file_path)
    metadata['extraction_method']='docling'
    metadata['raw_extraction_format']='markdown'
    log.info(f'Document successfully extracted with docling. ')
    return {'raw_text':raw_text,
            'cleaned_text': cleaned_text,
            'metadata': metadata}