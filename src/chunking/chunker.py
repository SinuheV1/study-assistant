from src.utils.logging import setup_logger
import re
import string
import hashlib
import os
import json


log = setup_logger(__name__)

def split_into_blocks(text:str) ->list[str]:
    """
    Split cleaned text into logical blocks using double newlines as separators.
    
    Args:
        text: Cleaned document text.
    Returns:
        List of non-empty text blocks.
    """
    #normalize tabs, repeated spaces, and blocks of blank lines
    normalized_lines = [line.rstrip() for line in text.splitlines()]
    normalized_text = "\n".join(normalized_lines)
    
    #split text into sections using double newline as primary seperator
    sections=re.split(r'\n\s*\n',normalized_text)
    blocks=[]
    
    for section in sections:
        #strip leading/trailing whitespace
        section=section.strip()
        #if section is empty continue, else append to blocks
        if not section:
            continue
        blocks.append(section)
        
    return blocks
    


def is_heading(block:str) ->bool:
    """
    Heuristic to detect whether a block is a section heading.
    A heading is short, non-empty, not a list item, and does not end
    with sentence-terminating punctuation.

    Args:
        block: A single text block.
    Returns:
        True if the block looks like a heading.
    """
    stripped_block = block.strip()

    if not stripped_block:
        return False

    list_pattern = r"^\d+[\.\)]\s+|^[-*]\s+"
    if re.match(list_pattern, stripped_block):
        return False

    if len(stripped_block) > 75:
        return False

    if stripped_block.endswith((".", "?", "!", ";", ":")):
        return False

    return True

def estimate_chunk_size(text:str)->int:
    """
    Estimate chunk size by raw character count, including whitespace.
    This aligns with how chunks are stored and retrieved, making
    target_size directly interpretable during tuning.

    Args:
        text: The chunk text to measure.
    Returns:
        Character count of the text as-is.
    """
    return len(text)

def split_into_sentences(text: str) -> list[str]:
    """
    Split a text block into individual sentences using punctuation boundaries.
    Handles common abbreviations and edge cases conservatively.
    Args:
        text: A text block to split.
    Returns:
        List of sentence strings.
    """
    #split on period/exclamation/question followed by whitespace + capital letter,
    #or end of string. Keeps the punctuation attached to the preceding sentence.
    raw = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences
def split_oversized_block(block: str, target_size: int, overlap_size: int) -> list[str]:
    """
    Sentence-level fallback splitter for blocks that exceed target_size.
    Accumulates sentences into sub-chunks up to target_size, then carries
    overlap forward at sentence granularity.

    Args:
        block: A text block larger than target_size.
        target_size: Maximum raw character count per chunk.
        overlap_size: Approximate character count to carry forward as overlap.
    Returns:
        List of sub-chunks derived from the block.
    """
    sentences = split_into_sentences(block)

    if not sentences:
        return [block]

    sub_chunks = []
    current = ""
    overlap_sentences: list[str] = []

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence

        if estimate_chunk_size(candidate) <= target_size:
            current = candidate
        else:
            if current:
                sub_chunks.append(current)
                # Build overlap from trailing sentences up to overlap_size chars
                overlap_sentences = _trailing_sentences(current, overlap_size)
                overlap_text = " ".join(overlap_sentences).strip()
                current = (overlap_text + " " + sentence).strip() if overlap_text else sentence
            else:
                # Single sentence exceeds target — emit as-is with a warning
                log.warning(
                    f"Single sentence exceeds target_size "
                    f"({estimate_chunk_size(sentence)} > {target_size}). "
                    "Emitting as oversized chunk."
                )
                sub_chunks.append(sentence)
                current = ""

    if current:
        sub_chunks.append(current)

    return sub_chunks

def _trailing_sentences(text: str, overlap_size: int) -> list[str]:
    """
    Return the trailing sentences of text whose combined length fits within
    overlap_size characters. Works backwards from the last sentence.

    Args:
        text: The chunk text to pull overlap from.
        overlap_size: Target overlap character budget.
    Returns:
        List of sentences (in forward order) that form the overlap.
    """
    if overlap_size <= 0:
        return []

    sentences = split_into_sentences(text)
    selected = []
    total = 0

    for sentence in reversed(sentences):
        total += len(sentence)
        if total > overlap_size and selected:
            break
        selected.insert(0, sentence)

    return selected
def _get_overlap(text: str, overlap_size: int) -> str:
    """
    Slice overlap text from the end of a chunk, snapping to the nearest
    word boundary to avoid cutting mid-word.

    Args:
        text: The completed chunk to pull overlap from.
        overlap_size: Target overlap character count.
    Returns:
        Overlap string starting at a word boundary.
    """
    if overlap_size <= 0 or len(text) <= overlap_size:
        return text

    tail = text[-overlap_size:]
    boundary = tail.find(" ")
    return tail[boundary:].strip() if boundary != -1 else tail


def _flush_chunk(current_chunk: str,new_block: str,overlap_size: int,chunks: list[str]) -> str:
    """
    Commit current_chunk to chunks, then start a new chunk seeded with
    overlap from the end of current_chunk followed by new_block.

    Args:
        current_chunk: The chunk being finalized.
        new_block: The block that triggered the flush (becomes start of next chunk).
        overlap_size: Character budget for overlap carry-forward.
        chunks: The list to append the finalized chunk to.
    Returns:
        The new current_chunk string (overlap + new_block).
    """
    chunks.append(current_chunk)
    overlap_text = _get_overlap(current_chunk, overlap_size)
    return (overlap_text + "\n\n" + new_block).strip() if overlap_text else new_block

def build_chunks_from_blocks(blocks: list[str],target_size: int,overlap_size: int) -> list[str]:
    """
    Assemble logical blocks into retrieval-friendly chunks. Attempts to keep
    headings attached to their following content. Blocks that exceed target_size
    are split at sentence boundaries before being added.

    Args:
        blocks: List of logical text blocks from split_into_blocks.
        target_size: Target maximum raw character count per chunk.
        overlap_size: Character budget to carry forward as overlap between chunks.
    Returns:
        List of chunk strings ready for embedding.
    """
    chunks: list[str] = []
    current_chunk = ""
    pending_heading = None

    for block in blocks:

        #expand oversized blocks into sub-blocks before processing
        if not is_heading(block) and estimate_chunk_size(block) > target_size:
            log.warning(
                f"Block exceeds target_size "
                f"({estimate_chunk_size(block)} > {target_size}). "
                "Applying sentence-level fallback split."
            )
            sub_blocks = split_oversized_block(block, target_size, overlap_size)
        else:
            sub_blocks = [block]

        for sub_block in sub_blocks:

            if is_heading(sub_block):
                if pending_heading is not None:
                    # Flush orphan heading into current chunk or start new one
                    candidate = (current_chunk + "\n\n" + pending_heading).strip()
                    if not current_chunk:
                        current_chunk = pending_heading
                    elif estimate_chunk_size(candidate) <= target_size:
                        current_chunk = candidate
                    else:
                        current_chunk = _flush_chunk(
                            current_chunk, pending_heading, overlap_size, chunks
                        )
                pending_heading = sub_block
                continue

            # Attach pending heading to this content block
            content = (pending_heading + "\n\n" + sub_block).strip() if pending_heading else sub_block
            pending_heading = None

            if not current_chunk:
                current_chunk = content
                continue

            candidate = current_chunk + "\n\n" + content

            if estimate_chunk_size(candidate) <= target_size:
                current_chunk = candidate
            else:
                current_chunk = _flush_chunk(
                    current_chunk, content, overlap_size, chunks
                )

    #flush remaining pending heading
    if pending_heading is not None:
        candidate = (current_chunk + "\n\n" + pending_heading).strip()
        if not current_chunk:
            current_chunk = pending_heading
        elif estimate_chunk_size(candidate) <= target_size:
            current_chunk = candidate
        else:
            current_chunk = _flush_chunk(current_chunk, pending_heading, overlap_size, chunks)

    #flush final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def create_chunk_metadata(document_metadata: dict,chunk_text: str,chunk_index: int,total_chunks: int) -> dict:
    """
    Create chunk-level metadata combining parent document metadata with
    chunk-specific fields. Chunk ID is a stable SHA-256 hash of
    document_id + chunk_text + chunk_index.

    Args:
        document_metadata: Metadata dict for the parent document.
        chunk_text: The text content of this chunk.
        chunk_index: Zero-based position of this chunk in the document.
        total_chunks: Total number of chunks produced from this document.
    Returns:
        Metadata dict for this chunk.
    """
    hash_object = hashlib.sha256()
    raw_str = f"{document_metadata['document_id']}|{chunk_text}|{chunk_index}"
    hash_object.update(raw_str.encode("utf-8"))
    chunk_id = "chunk_" + hash_object.hexdigest()[:12]
    
    return {
        "chunk_id": chunk_id,
        "document_id": document_metadata["document_id"],
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "source_type": document_metadata["source_type"],
        "course": document_metadata["course"],
        "title": document_metadata["title"],
        "file_path": document_metadata["file_path"],
        "chunk_text_length": len(chunk_text)}

def chunk_document(cleaned_text: str,document_metadata: dict,target_size: int,overlap_size: int) -> list[dict]:
    """
    Main entry point for turning one cleaned document into chunk records.
    Each record contains chunk_id, chunk_text, and metadata.

    Args:
        cleaned_text: Full cleaned text of the document.
        document_metadata: Metadata dict for the parent document.
        target_size: Target maximum raw character count per chunk.
        overlap_size: Character budget for overlap carry-forward.
    Returns:
        List of chunk record dicts, each with chunk_id, chunk_text, metadata.
    """
    blocks = split_into_blocks(cleaned_text)
    chunk_texts = build_chunks_from_blocks(blocks, target_size, overlap_size)
    total_chunks = len(chunk_texts)

    chunk_records = []

    for index, chunk_text in enumerate(chunk_texts):
        chunk_metadata = create_chunk_metadata(
            document_metadata=document_metadata,
            chunk_text=chunk_text,
            chunk_index=index,
            total_chunks=total_chunks)
        
        chunk_records.append({
            "chunk_id": chunk_metadata["chunk_id"],
            "chunk_text": chunk_text,
            "metadata": chunk_metadata})

    if not chunk_records:
        log.warning(f"No chunks produced for document '{document_metadata.get('document_id', 'unknown')}'. "
                    "Check that cleaned_text is non-empty.")

        return chunk_records

    log.info(f"Document '{chunk_records[0]['metadata']['document_id']}' "f"chunked into {total_chunks} chunks.")
    
    return chunk_records

def save_chunks(chunk_records: list[dict], output_dir: str) -> None:
    """
    Persist chunk records to a JSON file for inspection and later embedding.
    Output file is named <document_id>_chunks.json inside output_dir.

    Args:
        chunk_records: List of chunk record dicts from chunk_document.
        output_dir: Directory path where the output file will be written.
    """
    if not chunk_records:
        log.warning("No chunk records provided. Nothing was saved.")
        return

    os.makedirs(output_dir, exist_ok=True)

    document_id = chunk_records[0]["metadata"]["document_id"]
    output_file_name = f"{document_id}_chunks.json"
    full_output_path = os.path.join(output_dir, output_file_name)

    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(chunk_records)} chunk records to {full_output_path}")