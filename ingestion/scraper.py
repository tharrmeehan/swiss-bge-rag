import httpx
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
BASE = "https://www.bger.ch"
HEADERS = {"User-Agent": "swiss-bge-rag-research/1.0 (academic project)"}
CRAWL_DELAY = 1.5  # seconds between requests — be polite

# bger.ch serves individual leading decisions (BGE) at:
#   /ext/eurospider/live/de/php/clir/http/index.php?highlight_docid=atf://<vol>-<div>-<page>:<lang>&lang=de&type=show_document
# e.g. atf://148-I-1:de for "BGE 148 I 1". The page <title> holds the case number ("148 I 1").
SHOW_DOCUMENT_URL = (
    f"{BASE}/ext/eurospider/live/de/php/clir/http/index.php"
    "?highlight_docid=atf%3A%2F%2F{volume}-{division}-{page}%3Ade&lang=de&type=show_document"
)


def ruling_url(volume: int, division: str, page: int) -> str:
    """Build the direct show_document URL for a BGE ruling, e.g. volume=148, division='I', page=1."""
    return SHOW_DOCUMENT_URL.format(volume=volume, division=division, page=page)


def fetch_ruling_links(listing_url: str) -> list[str]:
    """Extract absolute ruling URLs from a BGer listing page."""
    resp = httpx.get(listing_url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "highlight_docid" in href and "show_document" in href:
            full = href if href.startswith("http") else BASE + href
            links.append(full)
    return list(dict.fromkeys(links))  # deduplicate, preserve order


def _case_number_from_html(html: str) -> str | None:
    """Extract the case number (e.g. "BGE 148 I 1") from a ruling page's <title>."""
    match = re.search(r"<title>\s*([\d]+ [IVX]+ \d+)\s*</title>", html)
    if not match:
        return None
    return f"BGE {match.group(1)}"


def fetch_ruling(url: str, case_number: str | None = None) -> Path:
    """Fetch a single ruling HTML and save to data/raw/<case_number>.html.

    If case_number is not given, it is extracted from the page <title>.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    html = resp.content.decode("latin-1")  # bger.ch serves ISO-8859-1

    resolved_case_number = case_number or _case_number_from_html(html)
    if not resolved_case_number:
        raise ValueError(f"could not determine case number for {url}")

    safe_name = resolved_case_number.replace(" ", "_").replace("/", "-")
    out = RAW_DIR / f"{safe_name}.html"
    if out.exists():
        return out  # already fetched
    out.write_text(html, encoding="utf-8")
    time.sleep(CRAWL_DELAY)
    return out


def scrape_division(listing_url: str, max_rulings: int = 500) -> list[Path]:
    """Scrape up to max_rulings from a division listing page."""
    links = fetch_ruling_links(listing_url)[:max_rulings]
    saved = []
    for i, url in enumerate(links):
        path = fetch_ruling(url)
        saved.append(path)
        print(f"[{i+1}/{len(links)}] {path.name}")
    return saved
