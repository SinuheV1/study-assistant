from __future__ import annotations

from src.ingestion import clean_text


def test_remove_image_artifacts_drops_image_lines_and_collapses_blanks():
    text = "Keep\n\n<!-- image -->\n\n\n[IMAGE]\n\nDone"

    assert clean_text.remove_image_artifacts(text) == "Keep\n\nDone"


def test_remove_html_comments_removes_inline_comments():
    assert clean_text.remove_html_comments("Before <!-- hidden --> after") == "Before  after"


def test_normalize_markdown_headers_adds_missing_space():
    assert clean_text.normalize_markdown_headers("##Title\n### Already") == "## Title\n### Already"


def test_fix_broken_line_wraps_joins_paragraphs_and_preserves_lists():
    text = "This line wraps\ninto the next\n\n1. Keep item\n- Keep bullet\nDone."

    assert clean_text.fix_broken_line_wraps(text) == (
        "This line wraps into the next\n\n1. Keep item\n- Keep bullet\nDone."
    )


def test_clean_short_noise_lines_removes_punctuation_comments_and_ocr_noise():
    text = "Keep me\n!!!\n<!-- image -->\nWhatareThedesired\nStill keep"

    assert clean_text.clean_short_noise_lines(text) == "Keep me\nStill keep"


def test_basic_text_cleaning_none_returns_empty_string():
    assert clean_text.basic_text_cleaning(None) == ""


def test_normalize_whitespace_and_trailing_whitespace():
    assert clean_text.remove_trailing_whitespace("a  \nb\t ") == "a\nb"
    assert clean_text.normalize_whitespace("a\t b   c\n\n\nnext") == "a b c\n\nnext"
