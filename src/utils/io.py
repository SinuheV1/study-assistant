import json
from pathlib import Path

from src.utils.logging import setup_logger

log = setup_logger(__name__)


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info(f"Saved json to {output_path}")


def save_extracted_text(document_record, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = document_record.get("metadata", {})
    document_id = metadata.get("document_id", "unknown_doc")
    output_file = output_dir / f"{document_id}_extracted_text.json"
    output_file = output_dir / f"{document_id}_extracted_text.json"

    data_to_save = {
        "metadata": metadata,
        "raw_text": document_record.get("raw_text", ""),
        "cleaned_text": document_record.get("cleaned_text", ""),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

    log.info(f"Saved extracted text for {document_id} to {output_file}")
