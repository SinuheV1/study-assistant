from src.ingestion.ingest_docling_document import ingest_docling_document

file_path = "data/raw/lecture_pdfs/Lecture_01.pdf"
result = ingest_docling_document(file_path)

if result is None:
    print("DOCILING INGESTION FAILED")
else:
    print("\n=== METADATA ===")
    print(result["metadata"])

    print("\n=== RAW TEXT PREVIEW ===")
    print(result["raw_text"][:2000])

    print("\n=== CLEANED TEXT PREVIEW ===")
    print(result["cleaned_text"][:2000])

    print("\n=== LENGTH CHECKS ===")
    print(f"Raw text length: {len(result['raw_text'])}")
    print(f"Cleaned text length: {len(result['cleaned_text'])}")