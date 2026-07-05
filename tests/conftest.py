from __future__ import annotations

import pytest


def _simple_sentence_splitter(text: str) -> list[str]:
    sentences = []

    for part in text.split("."):
        stripped = part.strip()
        if stripped:
            sentences.append(f"{stripped}.")

    return sentences


@pytest.fixture(autouse=True)
def stub_textbook_sentence_tokenizer(monkeypatch):
    from src.chunking import textbook_chunker

    monkeypatch.setattr(textbook_chunker, "_PUNKT_READY", True)
    monkeypatch.setattr(textbook_chunker, "ensure_nltk_punkt", lambda: None)
    monkeypatch.setattr(textbook_chunker, "sent_tokenize", _simple_sentence_splitter)
