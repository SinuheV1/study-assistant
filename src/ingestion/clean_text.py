import re
import string

from src.utils.logging import setup_logger

log = setup_logger(__name__)


def _weird_capitalization(text: str) -> bool:
    # checking for any lowercase character followed by uppercase
    if re.search(r"[a-z][A-Z]", text):
        return True
    # split words and check if end in lowercase but contains uppercase
    for word in text.split():
        if any(c.isupper() for c in word[1:]) and not word.isupper():
            return True
    return False


def remove_image_artifacts(text: str) -> str:
    cleaned_lines = []
    image_patterns = [r"^<!--\s*image\s*-->$", r"^\[image\]$", r"^\[IMAGE\]$"]
    for line in text.splitlines():
        stripped_line = line.strip()
        is_image_artifact = any(
            re.match(pattern, stripped_line, flags=re.IGNORECASE) for pattern in image_patterns
        )
        if is_image_artifact:
            continue
        cleaned_lines.append(line)
    # collapse repeated blank lines created after removals
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text


def remove_html_comments(text: str) -> str:
    pattern = "<!--.*?-->"
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def normalize_markdown_headers(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        normalized_line = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)
        cleaned_lines.append(normalized_line)
    return "\n".join(cleaned_lines)


def clean_short_noise_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            cleaned_lines.append("")
            continue
        # remove lines that are only punctuation
        punc_set = set(string.punctuation)
        if all(char in punc_set for char in stripped_line):
            continue
        # remove html/artifact images
        if "<!--" in stripped_line:
            continue
        if (
            " " not in stripped_line
            and len(stripped_line) > 15
            and _weird_capitalization(stripped_line)
        ):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def log_cleaning_stats(before_text: str, after_text: str) -> None:
    before_chars = len(before_text)
    after_chars = len(after_text)
    removed_chars = before_chars - after_chars
    log.info(f"Characters before cleaning: {before_chars}")
    log.info(f"Characters after cleaning: {after_chars}")
    log.info(f"Removed characters: {removed_chars}")


def normalize_whitespace(text: str) -> str:
    # normalize tabs, repeated spaces, and blocks of blank lines
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_trailing_whitespace(text: str) -> str:
    # clean line endings
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


def fix_broken_line_wraps(text: str) -> str:
    # joins lines that were split in transcripts or bad export
    lines = text.splitlines()
    cleaned_lines = []
    buffer = ""

    def is_list_item(line: str) -> bool:
        # list detection cleaning
        stripped = line.strip()
        if not stripped:
            return False

        if stripped.startswith(("- ", "* ")):
            return True

        # numbered-list detection: "1. item", "2. item", etc
        parts = stripped.split(maxsplit=1)
        if parts:
            token = parts[0]
            if token.endswith(".") and token[:-1].isdigit():
                return True

        return False

    for line in lines:
        stripped = line.strip()
        # blank line = paragraph/list boundary
        if not stripped:
            if buffer:
                cleaned_lines.append(buffer.strip())
                buffer = ""
            cleaned_lines.append("")
            continue
        # preserve list items on their own lines
        if is_list_item(stripped):
            if buffer:
                cleaned_lines.append(buffer.strip())
                buffer = ""
            cleaned_lines.append(stripped)
            continue

        # merge normal wrapped lines inside paragraphs
        if buffer and not buffer.endswith((".", ":", "?", "!")):
            buffer += " " + stripped
        else:
            if buffer:
                cleaned_lines.append(buffer.strip())
            buffer = stripped

    if buffer:
        cleaned_lines.append(buffer.strip())

    return "\n".join(cleaned_lines)


def basic_text_cleaning(text: str) -> str:
    if text is None:
        log.warning("Received None instead of text in basic_text_cleaning.")
        return ""
    original_text = text
    cleaned = remove_html_comments(text)
    cleaned = remove_image_artifacts(cleaned)
    cleaned = remove_trailing_whitespace(cleaned)
    cleaned = normalize_whitespace(cleaned)
    cleaned = normalize_markdown_headers(cleaned)
    cleaned = fix_broken_line_wraps(cleaned)
    cleaned = clean_short_noise_lines(cleaned)
    log_cleaning_stats(original_text, cleaned)
    log.info("Completed basic text cleaning.")
    return cleaned
