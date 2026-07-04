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
    """Extract the case number (e.g. "BGE 148 I 1") from a ruling page's <title>.

    Pre-reform divisions (Ia/Ib) render as all-caps "IA"/"IB" in the <title>;
    normalize back to the conventional "Ia"/"Ib" citation form.
    """
    match = re.search(r"<title>\s*(\d+) (I|II|III|IV|V|IA|IB) (\d+)\s*</title>", html)
    if not match:
        return None
    volume, division, page = match.groups()
    if division in ("IA", "IB"):
        division = division[0] + division[1].lower()
    return f"BGE {volume} {division} {page}"


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


# search.bger.ch serves a per-volume/division index of every leading decision:
#   index_atf.php?year=<volume>&volume=<I|II|III|IV|V>&lang=de&zoom=&system=clir
# "year" is actually the BGE volume number, not a calendar year. Divisions Ia/Ib
# (used pre-reform, e.g. "101 Ia 33") are grouped under volume=I in this index.
VOLUME_INDEX_URL = (
    "https://search.bger.ch/ext/eurospider/live/de/php/clir/http/index_atf.php"
    "?year={volume}&volume={division}&lang=de&zoom=&system=clir"
)
DIVISIONS = ("I", "II", "III", "IV", "V")


def _sample(items: list[str], cap: int) -> list[str]:
    """Evenly spaced sample of at most `cap` items, to spread across a volume's pages."""
    if len(items) <= cap:
        return items
    step = len(items) / cap
    return [items[int(i * step)] for i in range(cap)]


def scrape_volume_range(
    vol_min: int,
    vol_max: int,
    cap_per_division: int = 8,
    force_include: list[str] | None = None,
    divisions: tuple[str, ...] = DIVISIONS,
) -> list[Path]:
    """Sample rulings across BGE volumes vol_min..vol_max (inclusive).

    Takes up to `cap_per_division` evenly-spaced rulings per division per volume
    (keeps a full range scrape within a modest Qdrant free-tier footprint), plus
    always fetches any specific case numbers in `force_include`
    (e.g. ["139 I 218", "101 Ia 33"]) regardless of sampling.
    """
    saved = []
    for vol in range(vol_min, vol_max + 1):
        for division in divisions:
            listing = VOLUME_INDEX_URL.format(volume=vol, division=division)
            links = fetch_ruling_links(listing)
            for url in _sample(links, cap_per_division):
                path = fetch_ruling(url)
                saved.append(path)
                print(f"[{len(saved)}] {path.name}")

    for case_number in force_include or []:
        vol_str, division, page = case_number.split()
        url = ruling_url(int(vol_str), division, int(page))
        path = fetch_ruling(url, case_number=f"BGE {case_number}")
        saved.append(path)
        print(f"[{len(saved)}] {path.name} (forced)")

    return saved
