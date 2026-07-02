from dataclasses import dataclass
from bs4 import BeautifulSoup

SECTION_HEADING_IDS = {
    "sachverhalt": "Sachverhalt",
    "erwaegungen": "Erwägungen",
    "dispositiv": "Dispositiv",
}


@dataclass
class RulingSection:
    case_number: str
    year: int
    division: str
    section: str
    text: str


def _extract_division(case_number: str) -> str:
    # "BGE 148 I 1" -> "I"
    parts = case_number.strip().split()
    return parts[2] if len(parts) >= 3 else "Unknown"


def _extract_year(case_number: str) -> int:
    # BGE volume number -> approximate year (volume 148 ≈ 2022)
    # Volume 1 started in 1875; each volume is roughly one year
    parts = case_number.strip().split()
    try:
        volume = int(parts[1])
        return 1874 + volume
    except (IndexError, ValueError):
        return 0


def parse_ruling(html: str, case_number: str) -> list[RulingSection]:
    soup = BeautifulSoup(html, "html.parser")
    division = _extract_division(case_number)
    year = _extract_year(case_number)

    sections: list[RulingSection] = []
    current_label = "Unknown"
    current_paragraphs: list[str] = []

    def flush():
        if current_paragraphs:
            sections.append(RulingSection(
                case_number=case_number,
                year=year,
                division=division,
                section=current_label,
                text="\n\n".join(current_paragraphs),
            ))

    # bger.ch marks section headings as <span class="big bold" id="sachverhalt|erwaegungen|dispositiv">
    # and body paragraphs as <div class="paraatf">. Walk the document in order,
    # switching the current section on each heading span.
    for tag in soup.find_all(["span", "div"]):
        classes = tag.get("class") or []
        if tag.name == "span" and "big" in classes and "bold" in classes:
            heading_id = tag.get("id")
            label = SECTION_HEADING_IDS.get(heading_id)
            if label:
                flush()
                current_paragraphs = []
                current_label = label
            continue
        if tag.name == "div" and "paraatf" in classes:
            text = tag.get_text(separator=" ", strip=True)
            if text:
                current_paragraphs.append(text)

    flush()

    return sections
