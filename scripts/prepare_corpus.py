#!/usr/bin/env python3
"""
Prepare pretraining corpus from multiple sources.

Supports:
1. Chinese Wikipedia dump
2. CLUECorpus2020 (via HuggingFace datasets)
3. Custom text files

Output: Sharded JSONL files with 'text' field for HuggingFace datasets.
"""

import argparse
import bz2
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional
import multiprocessing as mp
from functools import partial

try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Chinese text cleaning patterns
PATTERNS_TO_REMOVE = [
    re.compile(r'\{\{[^}]+\}\}'),  # Templates
    re.compile(r'\[\[Category:[^\]]+\]\]'),  # Categories
    re.compile(r'\[\[File:[^\]]+\]\]'),  # Files
    re.compile(r'\[\[Image:[^\]]+\]\]'),  # Images
    re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL),  # References
    re.compile(r'<[^>]+>'),  # HTML tags
    re.compile(r'\[https?://[^\]]+\]'),  # External links
    re.compile(r'https?://\S+'),  # URLs
    re.compile(r'\s+'),  # Multiple whitespace
]


def clean_text(text: str) -> str:
    """Clean and normalize Chinese text."""
    for pattern in PATTERNS_TO_REMOVE:
        text = pattern.sub(' ', text)

    # Remove wiki links but keep text: [[link|text]] -> text
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()


def extract_wiki_articles(dump_path: Path) -> Iterator[dict]:
    """
    Extract articles from Wikipedia XML dump.

    Args:
        dump_path: Path to zhwiki-*-pages-articles.xml.bz2

    Yields:
        Dict with 'title' and 'text' fields
    """
    if not HAS_MWPARSER:
        raise ImportError("mwparserfromhell required for Wikipedia processing. "
                         "Install with: pip install mwparserfromhell")

    import xml.etree.ElementTree as ET

    # Namespace for Wikipedia XML
    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}

    print(f"Processing Wikipedia dump: {dump_path}")

    opener = bz2.open if str(dump_path).endswith('.bz2') else open

    with opener(dump_path, 'rt', encoding='utf-8') as f:
        # Iterative parsing for large files
        context = ET.iterparse(f, events=('end',))

        for event, elem in context:
            if elem.tag.endswith('page'):
                # Extract title and text
                title_elem = elem.find('.//mw:title', ns) or elem.find('title')
                text_elem = elem.find('.//mw:text', ns) or elem.find('.//text')

                if title_elem is not None and text_elem is not None:
                    title = title_elem.text or ''
                    wikitext = text_elem.text or ''

                    # Skip redirects, disambiguation, etc.
                    if wikitext.lower().startswith('#redirect'):
                        elem.clear()
                        continue

                    # Parse wikitext
                    try:
                        parsed = mwparserfromhell.parse(wikitext)
                        text = parsed.strip_code()
                        text = clean_text(text)

                        if len(text) > 100:  # Skip very short articles
                            yield {'title': title, 'text': text}
                    except Exception:
                        pass

                # Clear element to save memory
                elem.clear()


def load_clue_corpus(split: str = 'train') -> Iterator[dict]:
    """
    Load CLUECorpus2020 from HuggingFace.

    Args:
        split: Dataset split ('train')

    Yields:
        Dict with 'text' field
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")

    print(f"Loading CLUECorpus2020 ({split} split)...")

    try:
        dataset = load_dataset('clue', 'clue_corpus_small', split=split, streaming=True)
    except Exception:
        # Fallback to direct loading
        try:
            dataset = load_dataset('c4', 'zh', split=split, streaming=True)
        except Exception as e:
            print(f"Warning: Could not load CLUE corpus: {e}")
            print("Trying alternative Chinese corpus...")
            dataset = load_dataset('wikipedia', '20220301.zh', split='train', streaming=True)

    for item in dataset:
        text = item.get('text', '')
        if text and len(text) > 50:
            yield {'text': clean_text(text)}


def load_text_files(directory: Path) -> Iterator[dict]:
    """
    Load text files from a directory.

    Args:
        directory: Directory containing .txt files

    Yields:
        Dict with 'text' field
    """
    for filepath in directory.glob('**/*.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                text = clean_text(text)
                if len(text) > 50:
                    yield {'text': text, 'source': str(filepath)}
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")


def write_shards(
    data_iter: Iterator[dict],
    output_dir: Path,
    shard_size: int = 100000,
    prefix: str = 'corpus',
) -> int:
    """
    Write data to sharded JSONL files.

    Args:
        data_iter: Iterator of dicts
        output_dir: Output directory
        shard_size: Number of records per shard
        prefix: Filename prefix

    Returns:
        Total number of records written
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    shard_idx = 0
    shard_records = []

    for record in data_iter:
        shard_records.append(record)
        total += 1

        if len(shard_records) >= shard_size:
            shard_path = output_dir / f'{prefix}_{shard_idx:05d}.jsonl'
            with open(shard_path, 'w', encoding='utf-8') as f:
                for r in shard_records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"Wrote {shard_path} ({len(shard_records)} records)")
            shard_records = []
            shard_idx += 1

    # Write remaining records
    if shard_records:
        shard_path = output_dir / f'{prefix}_{shard_idx:05d}.jsonl'
        with open(shard_path, 'w', encoding='utf-8') as f:
            for r in shard_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Wrote {shard_path} ({len(shard_records)} records)")

    return total


def extract_unique_chars(output_dir: Path) -> set:
    """Extract unique CJK characters from corpus."""
    chars = set()

    for jsonl_path in output_dir.glob('*.jsonl'):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                text = record.get('text', '')
                for char in text:
                    cp = ord(char)
                    if (0x4E00 <= cp <= 0x9FFF or
                        0x3400 <= cp <= 0x4DBF or
                        0x20000 <= cp <= 0x2A6DF):
                        chars.add(char)

    return chars


def main():
    parser = argparse.ArgumentParser(
        description='Prepare pretraining corpus for CNM-BERT'
    )
    parser.add_argument(
        '--wiki-dump',
        type=Path,
        help='Path to Chinese Wikipedia dump (zhwiki-*-pages-articles.xml.bz2)'
    )
    parser.add_argument(
        '--use-clue-corpus',
        action='store_true',
        help='Include CLUECorpus2020 from HuggingFace'
    )
    parser.add_argument(
        '--text-dir',
        type=Path,
        help='Directory containing additional text files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/corpus'),
        help='Output directory (default: data/corpus)'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=100000,
        help='Number of records per shard (default: 100000)'
    )
    parser.add_argument(
        '--extract-chars',
        action='store_true',
        help='Extract unique characters after processing'
    )

    args = parser.parse_args()

    if not any([args.wiki_dump, args.use_clue_corpus, args.text_dir]):
        parser.error("At least one data source required: --wiki-dump, --use-clue-corpus, or --text-dir")

    # Process each source
    total_records = 0

    if args.wiki_dump:
        print("\n=== Processing Wikipedia ===")
        wiki_iter = extract_wiki_articles(args.wiki_dump)
        count = write_shards(
            wiki_iter,
            args.output_dir / 'wikipedia',
            args.shard_size,
            'wiki'
        )
        print(f"Wikipedia: {count} articles")
        total_records += count

    if args.use_clue_corpus:
        print("\n=== Processing CLUECorpus2020 ===")
        clue_iter = load_clue_corpus()
        count = write_shards(
            clue_iter,
            args.output_dir / 'clue',
            args.shard_size,
            'clue'
        )
        print(f"CLUECorpus: {count} records")
        total_records += count

    if args.text_dir:
        print(f"\n=== Processing text files from {args.text_dir} ===")
        text_iter = load_text_files(args.text_dir)
        count = write_shards(
            text_iter,
            args.output_dir / 'custom',
            args.shard_size,
            'custom'
        )
        print(f"Custom text: {count} records")
        total_records += count

    print(f"\n=== Summary ===")
    print(f"Total records: {total_records}")
    print(f"Output directory: {args.output_dir}")

    # Extract unique characters
    if args.extract_chars:
        print("\nExtracting unique CJK characters...")
        chars = extract_unique_chars(args.output_dir)
        chars_path = args.output_dir / 'unique_chars.txt'
        with open(chars_path, 'w', encoding='utf-8') as f:
            for char in sorted(chars):
                f.write(char + '\n')
        print(f"Found {len(chars)} unique CJK characters")
        print(f"Saved to: {chars_path}")


if __name__ == '__main__':
    main()
