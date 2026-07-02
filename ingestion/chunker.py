from dataclasses import dataclass
import tiktoken
from ingestion.parser import RulingSection

enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    text: str
    case_number: str
    year: int
    division: str
    section: str
    chunk_index: int


def chunk_section(section: RulingSection, max_tokens: int = 400, overlap_paragraphs: int = 1) -> list[Chunk]:
    paragraphs = [p.strip() for p in section.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append(_make_chunk(section, "\n\n".join(current), chunk_index))
            chunk_index += 1
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_tokens = sum(len(enc.encode(p)) for p in current)
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append(_make_chunk(section, "\n\n".join(current), chunk_index))

    return chunks


def _make_chunk(section: RulingSection, text: str, index: int) -> Chunk:
    chunk_id = f"{section.case_number.replace(' ', '_')}_{section.section}_{index}"
    return Chunk(
        id=chunk_id,
        text=text,
        case_number=section.case_number,
        year=section.year,
        division=section.division,
        section=section.section,
        chunk_index=index,
    )
