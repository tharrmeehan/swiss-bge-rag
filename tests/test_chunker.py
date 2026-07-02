import pytest
from ingestion.chunker import chunk_section, Chunk
from ingestion.parser import RulingSection


def _make_section(text: str) -> RulingSection:
    return RulingSection(case_number="BGE 148 I 1", year=2022, division="I", section="Erwägungen", text=text)


def test_short_section_is_single_chunk():
    section = _make_section("Kurzer Text.")
    chunks = chunk_section(section, max_tokens=400)
    assert len(chunks) == 1
    assert chunks[0].text == "Kurzer Text."


def test_long_section_is_split():
    para = "Dies ist ein Absatz mit etwas Text. " * 30  # ~200 tokens per repeat
    section = _make_section(f"{para}\n\n{para}\n\n{para}")
    chunks = chunk_section(section, max_tokens=400)
    assert len(chunks) >= 2


def test_chunk_inherits_metadata():
    section = _make_section("Text.")
    chunks = chunk_section(section, max_tokens=400)
    assert chunks[0].case_number == "BGE 148 I 1"
    assert chunks[0].division == "I"
    assert chunks[0].section == "Erwägungen"
    assert chunks[0].chunk_index == 0


def test_chunk_ids_are_unique():
    para = "Absatz. " * 60
    section = _make_section(f"{para}\n\n{para}\n\n{para}")
    chunks = chunk_section(section, max_tokens=400)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
