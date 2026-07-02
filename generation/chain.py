import os
from openai import OpenAI
from retrieval.retriever import SearchResult
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = """Du bist ein juristischer Assistent, der Fragen zu Schweizer Bundesgerichtsentscheiden (BGE) beantwortet.

Regeln:
- Beantworte die Frage ausschliesslich auf Basis der bereitgestellten Quellen.
- Zitiere für jede sachliche Aussage die Fallnummer in Klammern, z.B. (BGE 148 I 1).
- Wenn die Antwort nicht in den Quellen enthalten ist, antworte genau: "Diese Information ist in den abgerufenen Entscheiden nicht enthalten."
- Antworte auf Deutsch."""


def format_sources(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Quelle {i} — {r.case_number}, {r.section}]\n{r.text}")
    return "\n\n".join(parts)


def answer(query: str, results: list[SearchResult], model: str = "gpt-4o") -> str:
    sources_text = format_sources(results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Quellen:\n\n{sources_text}\n\nFrage: {query}"},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content
