"""Scrape Suits episode scripts from Springfield-Springfield."""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BASE_URL = "https://www.springfieldspringfield.co.uk"
EPISODE_URL = BASE_URL + "/view_episode_scripts.php?tv-show=suits&episode={episode}"
REQUEST_DELAY_SEC = 2.0


def list_episodes(seasons: range | None = None) -> list[str]:
    """Return episode slugs like s01e01, s01e02, ... for configured seasons."""
    if seasons is None:
        seasons = range(1, 8)  # Suits seasons 1-7

    episodes: list[str] = []
    for season in seasons:
        # Most Suits seasons have 12-16 episodes; scan up to 16
        for ep in range(1, 17):
            episodes.append(f"s{season:02d}e{ep:02d}")
    return episodes


def fetch_episode_script(episode: str) -> str | None:
    """Fetch raw script text for one episode (e.g. s01e01). Returns None if not found."""
    url = EPISODE_URL.format(episode=episode)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    # Script body starts after the episode title line (e.g. "Pilot")
    lines = text.split("\n")
    start_idx = _find_script_start(lines)
    if start_idx is None:
        return None

    script_lines = [line.strip() for line in lines[start_idx:] if line.strip()]
    if len(script_lines) < 50:
        return None

    return "\n".join(script_lines)


def scrape_seasons(
    output_dir: Path,
    seasons: range | None = None,
    delay_sec: float = REQUEST_DELAY_SEC,
) -> list[Path]:
    """Download episode scripts and save to data/raw/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for episode in list_episodes(seasons):
        out_path = output_dir / f"{episode}.txt"
        if out_path.exists() and out_path.stat().st_size > 1000:
            saved.append(out_path)
            continue

        script = fetch_episode_script(episode)
        if not script:
            continue

        header = f"# Episode: {episode.upper()}\n"
        out_path.write_text(header + script, encoding="utf-8")
        saved.append(out_path)
        time.sleep(delay_sec)

    return saved


def _find_script_start(lines: list[str]) -> int | None:
    """Locate where dialogue begins after page chrome and episode title."""
    title_pattern = re.compile(r"^Suits s\d{2}e\d{2} Episode Script$", re.I)
    for i, line in enumerate(lines):
        if title_pattern.match(line.strip()):
            # Next non-empty line is episode title; dialogue follows
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].strip() and not lines[j].startswith(">"):
                    return j + 1
    return None
