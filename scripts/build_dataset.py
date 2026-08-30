#!/usr/bin/env python3
"""CLI for building the Harvey Specter persona dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Allow running as `python scripts/build_dataset.py` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.dataset_config import (
    DEFAULT_GEMINI_MODEL,
    FINAL_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SYNTHETIC_DIR,
)
from dataset_builder.extractors.harvey_extractor import deduplicate_pairs, extract_pairs
from dataset_builder.formatters.chat_formatter import (
    load_pairs_jsonl,
    merge_and_split,
    save_jsonl,
)
from dataset_builder.generators.synthetic_generator import (
    generate_synthetic_pairs,
    load_seed_prompts,
)
from dataset_builder.parsers.subtitle import parse_directory
from dataset_builder.parsers.transcript_scraper import fetch_transcript, scrape_and_save


@click.group()
def cli() -> None:
    """Harvey-LLM persona dataset builder."""
    load_dotenv()


@cli.command("parse")
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), default=RAW_DIR)
@click.option("--output", type=click.Path(path_type=Path), default=PROCESSED_DIR / "harvey_pairs.jsonl")
def parse_cmd(input_dir: Path, output: Path) -> None:
    """Parse raw transcripts/subtitles and extract Harvey context-response pairs."""
    click.echo(f"Parsing files in {input_dir}...")
    lines = parse_directory(input_dir)
    click.echo(f"  Found {len(lines)} dialogue lines")

    pairs = extract_pairs(lines)
    pairs = deduplicate_pairs(pairs)
    click.echo(f"  Extracted {len(pairs)} Harvey pairs")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")

    click.echo(f"Saved to {output}")


@cli.command("scrape")
@click.argument("url")
@click.option("--output-dir", type=click.Path(path_type=Path), default=RAW_DIR)
@click.option("--episode-name", default=None, help="Optional episode label for the saved file")
def scrape_cmd(url: str, output_dir: Path, episode_name: str | None) -> None:
    """Scrape a single fan transcript URL and save to data/raw/."""
    from dataset_builder.parsers.transcript_scraper import _slug_from_url

    output_dir.mkdir(parents=True, exist_ok=True)
    name = episode_name or _slug_from_url(url)
    out_path = output_dir / f"{name}.txt"

    click.echo(f"Fetching {url}...")
    lines = fetch_transcript(url, episode_name=name)
    if not lines:
        click.echo("No dialogue found. Check the URL or page format.", err=True)
        raise SystemExit(1)

    content = "\n".join(
        f"{line.speaker}: {line.text}" if line.speaker else line.text for line in lines
    )
    out_path.write_text(content, encoding="utf-8")
    click.echo(f"Saved {len(lines)} lines to {out_path}")


@cli.command("scrape-batch")
@click.argument("index_url")
@click.option("--output-dir", type=click.Path(path_type=Path), default=RAW_DIR)
@click.option("--max-pages", default=10, help="Max episode pages to fetch")
def scrape_batch_cmd(index_url: str, output_dir: Path, max_pages: int) -> None:
    """Scrape multiple transcript URLs from an index page."""
    from dataset_builder.parsers.transcript_scraper import fetch_transcript_list

    urls = fetch_transcript_list(index_url, max_pages=max_pages)
    click.echo(f"Found {len(urls)} transcript URLs")
    saved = scrape_and_save(urls, output_dir)
    click.echo(f"Saved {len(saved)} files to {output_dir}")


@cli.command("generate")
@click.option("--prompts", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=SYNTHETIC_DIR / "synthetic_pairs.jsonl")
@click.option("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for generation")
@click.option("--count", default=0, help="Limit number of prompts (0 = all)")
def generate_cmd(prompts: Path | None, output: Path, model: str, count: int) -> None:
    """Generate synthetic Harvey-style pairs using a frontier LLM."""
    seed_prompts = load_seed_prompts(prompts)
    if count > 0:
        seed_prompts = seed_prompts[:count]

    click.echo(f"Generating {len(seed_prompts)} synthetic pairs with {model}...")
    if output.exists():
        output.unlink()

    pairs = generate_synthetic_pairs(seed_prompts, model=model, output_path=output)
    click.echo(f"Generated {len(pairs)} pairs → {output}")


@cli.command("merge")
@click.option("--transcript", type=click.Path(exists=True, path_type=Path), default=PROCESSED_DIR / "harvey_pairs.jsonl")
@click.option("--synthetic", type=click.Path(path_type=Path), default=SYNTHETIC_DIR / "synthetic_pairs.jsonl")
@click.option("--output-dir", type=click.Path(path_type=Path), default=FINAL_DIR)
@click.option("--val-ratio", default=0.1, help="Validation split ratio")
def merge_cmd(
    transcript: Path,
    synthetic: Path,
    output_dir: Path,
    val_ratio: float,
) -> None:
    """Merge transcript + synthetic pairs into train/val JSONL for fine-tuning."""
    transcript_pairs = load_pairs_jsonl(transcript) if transcript.exists() else []
    synthetic_pairs = load_pairs_jsonl(synthetic) if synthetic.exists() else []

    click.echo(f"Transcript pairs: {len(transcript_pairs)}")
    click.echo(f"Synthetic pairs:  {len(synthetic_pairs)}")

    train, val = merge_and_split(transcript_pairs, synthetic_pairs, val_ratio=val_ratio)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)

    stats = {
        "train_count": len(train),
        "val_count": len(val),
        "transcript_count": len(transcript_pairs),
        "synthetic_count": len(synthetic_pairs),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    click.echo(f"Train: {len(train)} → {train_path}")
    click.echo(f"Val:   {len(val)} → {val_path}")


@cli.command("build-all")
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), default=RAW_DIR)
@click.option("--generate-synthetic/--no-generate-synthetic", default=False)
@click.option("--model", default=DEFAULT_GEMINI_MODEL)
def build_all_cmd(input_dir: Path, generate_synthetic: bool, model: str) -> None:
    """Run the full pipeline: parse → (optional) generate → merge."""
    ctx = click.get_current_context()

    pairs_path = PROCESSED_DIR / "harvey_pairs.jsonl"
    ctx.invoke(parse_cmd, input_dir=input_dir, output=pairs_path)

    if generate_synthetic:
        ctx.invoke(
            generate_cmd,
            prompts=None,
            output=SYNTHETIC_DIR / "synthetic_pairs.jsonl",
            model=model,
            count=0,
        )

    ctx.invoke(
        merge_cmd,
        transcript=pairs_path,
        synthetic=SYNTHETIC_DIR / "synthetic_pairs.jsonl",
        output_dir=FINAL_DIR,
        val_ratio=0.1,
    )


if __name__ == "__main__":
    cli()
