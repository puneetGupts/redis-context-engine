"""
Scrapes Redis documentation from redis.io/docs.

Pain point this solves:
  We need a knowledge base to search over. Redis's own docs are the perfect
  corpus — publicly available, well-structured, and directly relevant to the
  domain. Scraping them once and indexing into Redis means all future queries
  hit the vector store, not the web.
"""

import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Generator
from urllib.parse import urljoin, urlparse

# Redis doc sections most relevant to Context Engine use cases
SEED_URLS = [
    "https://redis.io/docs/latest/develop/data-types/vector-sets/",
    "https://redis.io/docs/latest/develop/data-types/vector-sets/filtered-search/",
    "https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/",
    "https://redis.io/docs/latest/develop/interact/search-and-query/",
    "https://redis.io/docs/latest/develop/interact/search-and-query/query/",
    "https://redis.io/docs/latest/develop/data-types/",
    "https://redis.io/docs/latest/develop/data-types/strings/",
    "https://redis.io/docs/latest/develop/data-types/hashes/",
    "https://redis.io/docs/latest/develop/use/pipelining/",
    "https://redis.io/docs/latest/develop/data-types/streams/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; redis-context-engine/1.0; educational project)"
}


@dataclass
class ScrapedDocument:
    url: str
    title: str
    content: str
    section: str


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract clean text from a documentation page."""
    # Remove nav, footer, sidebar — we only want the article content
    for tag in soup.select("nav, footer, aside, .sidebar, .header, script, style"):
        tag.decompose()

    article = soup.select_one("article, main, .content, #main-content")
    if article:
        return article.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def _get_section(url: str) -> str:
    """Derive a human-readable section label from the URL."""
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 3:
        return " / ".join(parts[2:4]).replace("-", " ").title()
    return "Redis Docs"


def _get_doc_links(url: str, soup: BeautifulSoup) -> list[str]:
    """Find internal doc links on the page to follow."""
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"])
        parsed = urlparse(href)
        if (
            parsed.netloc == "redis.io"
            and "/docs/" in parsed.path
            and not parsed.path.endswith(".pdf")
        ):
            # Strip query params AND fragments — anchors (#section) are the same page
            clean = href.split("?")[0].split("#")[0]
            if clean:
                links.append(clean)
    return list(set(links))


def scrape_docs(max_pages: int = 50, delay: float = 1.0) -> Generator[ScrapedDocument, None, None]:
    """
    Scrape Redis documentation pages starting from seed URLs.

    Args:
        max_pages: Maximum number of pages to scrape (keeps it manageable for free tier).
        delay: Seconds to wait between requests (be a good citizen).

    Yields:
        ScrapedDocument for each successfully scraped page.
    """
    visited: set[str] = set()
    queue: list[str] = list(SEED_URLS)
    scraped = 0

    print(f"Starting scrape — target: {max_pages} pages")

    while queue and scraped < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  Skipping {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        content = _extract_text(soup)
        if len(content) < 200:
            # Too short — likely a redirect or empty page
            continue

        doc = ScrapedDocument(
            url=url,
            title=title,
            content=content,
            section=_get_section(url),
        )
        yield doc
        scraped += 1
        print(f"  [{scraped}/{max_pages}] {title[:60]} ({url})")

        # Discover more pages from this one
        new_links = _get_doc_links(url, soup)
        for link in new_links:
            if link not in visited:
                queue.append(link)

        time.sleep(delay)

    print(f"Scrape complete — {scraped} pages collected")
