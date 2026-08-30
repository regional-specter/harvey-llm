"""Scrape fan transcript sites for Suits episode scripts."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from dataset_builder.models import DialogueLine

USER_AGENT = (
    "Mozilla/5.0 (compatible; harvey-llm-dataset-builder/0.1; +https://github.com/)"
)
REQUEST_DELAY_SEC = 1.5


def fetch_transcript(url: str, episode_name: str | None = None) -> list[DialogueLine]:
    """Fetch and parse a transcript page from a fan transcript site.

    Expects pages where dialogue appears as plain text with "SPEAKER: line" format,
    or inside <p> / <div> tags with that pattern.
    """
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    content = soup.find("article") or soup.find("main") or soup.find("body")
    if content is None:
        return []

    source_name = episode_name or _slug_from_url(url)
    lines: list[DialogueLine] = []

    for i, block in enumerate(content.stripped_strings):
        text = str(block).strip()
        if _looks_like_dialogue(text):
            speaker, dialogue = _split_dialogue(text)
            if dialogue:
                lines.append(
                    DialogueLine(
                        speaker=speaker,
                        text=dialogue,
                        source_file=f"{source_name}.web",
                        line_index=i,
                    )
                )

    return lines


def fetch_transcript_list(base_url: str, max_pages: int = 10) -> list[str]:
    """Discover episode transcript URLs from an index page."""
    response = requests.get(base_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        if _looks_like_transcript_link(full_url, anchor.get_text()):
            seen.add(full_url)
            urls.append(full_url)
        if len(urls) >= max_pages:
            break

    return urls


def scrape_and_save(
    urls: list[str],
    output_dir: Path,
    delay_sec: float = REQUEST_DELAY_SEC,
) -> list[Path]:
    """Fetch multiple transcript URLs and save as .txt files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for url in urls:
        slug = _slug_from_url(url)
        out_path = output_dir / f"{slug}.txt"

        if out_path.exists():
            saved.append(out_path)
            continue

        lines = fetch_transcript(url, episode_name=slug)
        if not lines:
            continue

        content = "\n".join(
            f"{line.speaker}: {line.text}" if line.speaker else line.text
            for line in lines
        )
        out_path.write_text(content, encoding="utf-8")
        saved.append(out_path)
        time.sleep(delay_sec)

    return saved


def _looks_like_dialogue(text: str) -> bool:
    return bool(re.match(r"^[A-Z][A-Za-z.\s']{0,30}:\s+\S", text))


def _split_dialogue(text: str) -> tuple[str | None, str | None]:
    match = re.match(r"^\s*(?P<speaker>[A-Z][A-Za-z.\s']+?)\s*:\s*(?P<text>.+)$", text)
    if match:
        return match.group("speaker").strip(), match.group("text").strip()
    return None, None


def _looks_like_transcript_link(url: str, link_text: str) -> bool:
    lower_url = url.lower()
    lower_text = link_text.lower()
    keywords = ("transcript", "script", "suits", "episode", "season")
    return any(k in lower_url or k in lower_text for k in keywords)


def _slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] or "episode"
    return re.sub(r"[^\w\-]", "_", slug)[:80]
