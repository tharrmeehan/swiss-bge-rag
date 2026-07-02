import pytest
from pathlib import Path
from ingestion.parser import parse_ruling, RulingSection

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ruling.html"


def test_parse_returns_sections():
    html = FIXTURE.read_text(encoding="utf-8")
    sections = parse_ruling(html, case_number="BGE 148 I 1")
    assert len(sections) > 0


def test_sections_have_required_fields():
    html = FIXTURE.read_text(encoding="utf-8")
    sections = parse_ruling(html, case_number="BGE 148 I 1")
    for s in sections:
        assert s.case_number == "BGE 148 I 1"
        assert s.division == "I"
        assert s.section in ("Sachverhalt", "Erwägungen", "Dispositiv", "Unknown")
        assert len(s.text) > 0


def test_erwagungen_is_longest_section():
    html = FIXTURE.read_text(encoding="utf-8")
    sections = parse_ruling(html, case_number="BGE 148 I 1")
    by_label = {s.section: s for s in sections}
    if "Erwägungen" in by_label:
        assert len(by_label["Erwägungen"].text) >= len(by_label.get("Dispositiv", RulingSection("", 0, "", "", "")).text)
