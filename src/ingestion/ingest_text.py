import datetime
import hashlib
import os
from pathlib import Path

from src.ingestion.clean_text import basic_text_cleaning
from src.utils.logging import setup_logger

log = setup_logger(__name__)


def detect_text_file_type(file_path):
    file_path = str(file_path)
    # allowed extensions
    allowed_extensions = [".txt", ".md"]
    extension = os.path.splitext(file_path)[1].lower()
    if extension in allowed_extensions:
        return extension
    log.critical(f"Unsupported file type specified at path: {file_path}")
    return None


def read_text_file(file_path):
    # read text file as a single string
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        log.critical(f"File does not exist at path: {file_path}")
    except PermissionError:
        log.critical(f"Permission denied. Unable to access file at path: {file_path}")
    except Exception as e:
        log.critical(f"Unexpected error occured while reading {file_path}: {e}")
    return None


def infer_course_from_path(file_path):
    # normalize file path
    normalized_path = os.path.normpath(file_path)
    components = normalized_path.split(os.sep)
    # filter out empty strings
    components = [component for component in components if component]
    search_string = "raw"
    try:
        # return index for 'raw' in file path
        index = components.index(search_string)
        log.info(f"Found {search_string} at index: {index}")
    except ValueError:
        log.warning(
            f"Search string: {search_string} not found in list. No course label will be attached."
        )
        return None
    # course should be 2 index after 'raw' always
    course_index = index + 2
    # need at least: ... / raw / source_type / course / file
    if len(components) <= course_index:
        log.info("No course folder found after source_type.")
        return None

    # if course_index points to the final path part, it's the filename, not a course
    if course_index == len(components) - 1:
        log.info("Path does not include a course folder; file is directly under source_type.")
        return None

    course = components[course_index]
    log.info(f"Found course '{course}' in path.")
    return course


def infer_source_from_path(file_path):
    # normalize file path
    normalized_path = os.path.normpath(file_path)
    components = normalized_path.split(os.sep)
    # filter out empty strings
    components = [component for component in components if component]
    search_string = "raw"
    try:
        # return index for 'raw' in file path
        index = components.index(search_string)
        log.info(f"Found {search_string} at index: {index}")
    except ValueError:
        log.warning(
            f"Search string: {search_string} not found in list. No source type label will be attached."
        )
        return None

    # source type will be 1 index after 'raw' always
    source_index = index + 1
    # need at least: ... / raw / source_type / course / file
    if len(components) <= source_index:
        log.info("No source folder found after source_type.")
        return None

    # if source_index points to the final path part, it's the filename, not a course
    if source_index == len(components) - 1:
        log.info("Path does not include a source folder; file is directly under source_type.")
        return None

    source_type = components[source_index]
    log.info(f"Found source '{source_type}' in path.")
    return source_type


def normalize_source_type(source_type):
    if source_type is None:
        return "generic_text"

    source_type = source_type.strip().lower()

    source_type_map = {
        "lecture_pdfs": "lecture_pdfs",
        "lecture_pdf": "lecture_pdfs",
        "lectures": "lecture_pdfs",
        "textbooks": "textbook_pdf",
        "textbook": "textbook_pdf",
        "textbook_pdfs": "textbook_pdf",
        "textbook_pdf": "textbook_pdf",
        "youtube": "youtube_transcript",
        "youtube_transcripts": "youtube_transcript",
        "transcripts": "youtube_transcript",
        "notes": "personal_notes",
        "personal_notes": "personal_notes",
        "markdown_notes": "personal_notes",
        "research_papers": "research_paper",
        "papers": "research_paper",
    }

    return source_type_map.get(source_type, source_type)


def build_document_metadata(file_path):
    file_path = Path(file_path)
    file_path_str = file_path.as_posix()

    course = infer_course_from_path(file_path)
    raw_source_type = infer_source_from_path(file_path)
    source_type = normalize_source_type(raw_source_type)
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    title = os.path.splitext(file_name)[0]
    extension = os.path.splitext(file_path)[1].lower()
    now = datetime.datetime.now()
    hash_object = hashlib.sha256(file_path_str.encode("utf-8"))
    document_id = "doc_" + hash_object.hexdigest()[:12]

    metadata = {
        "document_id": document_id,
        "file_name": file_name,
        "file_path": file_path_str,
        "source_type": source_type,
        "raw_source_type": raw_source_type,
        "title": title,
        "course": course,
        "topic": None,
        "ingestion_timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "file_type": extension,
        "file_size": file_size,
    }

    return metadata


def ingest_text_document(file_path):
    log.info(f"Starting ingestion for file: {file_path}")
    file_type = detect_text_file_type(file_path)
    if file_type is None:
        log.critical("Stopping ingestion: unsupported file type.")
        return None
    text_content = read_text_file(file_path)
    if text_content is None:
        log.critical("Stopping ingestion: file could not be read.")
        return None
    cleaned_text = basic_text_cleaning(text_content)
    metadata = build_document_metadata(file_path)
    log.info("Successfully built document metadata.")
    log.info(f"Document ID: {metadata['document_id']}")
    log.info(f"Source type: {metadata['source_type']}")
    log.info(f"Course: {metadata['course']}")
    log.info(f"File type: {metadata['file_type']}")
    log.info(f"File size: {metadata['file_size']} bytes")

    return {
        "file_type": file_type,
        "raw_text": text_content,
        "cleaned_text": cleaned_text,
        "metadata": metadata,
    }
