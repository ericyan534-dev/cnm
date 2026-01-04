#!/usr/bin/env python3
"""
Prepare IDS data and build CNM vocabulary.

This script:
1. Loads downloaded IDS data
2. Parses all characters into trees
3. Builds component/operator vocabularies
4. Saves CNM vocabulary for training
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cnm.data.ids_parser import IDSParser, estimate_ids_coverage
from cnm.data.vocab import CNMVocab, extract_corpus_chars


#!/usr/bin/env python3
"""
Prepare IDS data and build CNM vocabulary.

This script:
1. Loads downloaded IDS data
2. Parses all characters into trees
3. Builds component/operator vocabularies
4. Saves CNM vocabulary for training
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cnm.data.ids_parser import IDSParser, estimate_ids_coverage
from cnm.data.vocab import CNMVocab, extract_corpus_chars


def load_corpus_chars(corpus_dir: Path) -> Set[str]:
    """Load unique characters from corpus files."""
    chars: Set[str] = set()

    for jsonl_path in corpus_dir.glob('**/*.jsonl'):
        print(f"  Loading characters from {jsonl_path.name}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get('text', '')
                    for char in text:
                        cp = ord(char)
                        if (0x4E00 <= cp <= 0x9FFF or
                            0x3400 <= cp <= 0x4DBF or
                            0x20000 <= cp <= 0x2A6DF or
                            0x2F00 <= cp <= 0x2FDF or
                            0x2E80 <= cp <= 0x2EFF):
                            chars.add(char)
                except json.JSONDecodeError:
                    continue

    return chars


def load_bert_vocab_chars(vocab_file: Path) -> Set[str]:
    """Load CJK characters from BERT vocabulary file."""
    chars: Set[str] = set()

    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if len(token) == 1:
                cp = ord(token)
                if (0x4E00 <= cp <= 0x9FFF or
                    0x3400 <= cp <= 0x4DBF or
                    0x20000 <= cp <= 0x2A6DF):
                    chars.add(token)

    return chars


def main():
    parser = argparse.ArgumentParser(
        description='Prepare IDS data and build CNM vocabulary'
    )
    parser.add_argument(
        '--ids-file',
        type=Path,
        default=Path('data/ids/ids_parsed.json'),
        help='Path to parsed IDS JSON file'
    )
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        help='Directory containing corpus JSONL files (optional)'
    )
    parser.add_argument(
        '--bert-vocab',
        type=Path,
        help='Path to BERT vocab.txt file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/ids'),
        help='Output directory for CNM vocabulary'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--min-component-freq',
        type=int,
        default=1,
        help='Minimum frequency for components in vocabulary'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load IDS data
    print(f"Loading IDS data from {args.ids_file}...")
    if not args.ids_file.exists():
        print(f"Error: IDS file not found: {args.ids_file}")
        print("Run scripts/download_ids.py first to download IDS data.")
        sys.exit(1)

    with open(args.ids_file, 'r', encoding='utf-8') as f:
        ids_data = json.load(f)
    print(f"Loaded {len(ids_data)} characters with IDS decompositions")

    # Create parser
    parser_instance = IDSParser(ids_data=ids_data, max_depth=args.max_depth)

    # Collect target characters
    target_chars: Set[str] = set()

    # From IDS data
    target_chars.update(ids_data.keys())
    print(f"Characters from IDS: {len(target_chars)}")

    # From corpus
    if args.corpus_dir and args.corpus_dir.exists():
        print(f"\nLoading characters from corpus: {args.corpus_dir}")
        corpus_chars = load_corpus_chars(args.corpus_dir)
        print(f"Characters from corpus: {len(corpus_chars)}")
        target_chars.update(corpus_chars)

    # From BERT vocab
    if args.bert_vocab and args.bert_vocab.exists():
        print(f"\nLoading characters from BERT vocab: {args.bert_vocab}")
        bert_chars = load_bert_vocab_chars(args.bert_vocab)
        print(f"Characters from BERT vocab: {len(bert_chars)}")
        target_chars.update(bert_chars)

    print(f"\nTotal target characters: {len(target_chars)}")

    # Parse all characters
    print("\nParsing characters...")
    parser_instance.parse_all(target_chars)
    print(f"Parsed {parser_instance.num_cached} characters")

    # Compute coverage
    decomposable, total, coverage = estimate_ids_coverage(parser_instance, target_chars)
    print(f"\nIDS Coverage: {decomposable}/{total} = {coverage:.1%}")

    # Compute depth distribution
    depth_dist = parser_instance.compute_depth_distribution()
    print("\nTree depth distribution:")
    for depth, count in sorted(depth_dist.items()):
        print(f"  Depth {depth}: {count}")

    # Build vocabulary
    print("\nBuilding CNM vocabulary...")
    vocab = CNMVocab.build(
        parser_instance,
        target_chars,
        min_component_freq=args.min_component_freq,
    )

    print(f"\nVocabulary statistics:")
    print(f"  Components: {vocab.component_vocab_size}")
    print(f"  Operators: {vocab.operator_vocab_size}")
    print(f"  Characters with trees: {len(vocab.char_to_indexed)}")

    # Save vocabulary
    vocab_path = args.output_dir / 'cnm_vocab.json'
    vocab.save(vocab_path)
    print(f"\nSaved CNM vocabulary to {vocab_path}")

    # Save parser cache (for faster loading)
    cache_path = args.output_dir / 'ids_cache.json'
    parser_instance.save_cache(cache_path)
    print(f"Saved parse cache to {cache_path}")

    # Save statistics
    stats = {
        'total_characters': len(target_chars),
        'decomposable_characters': decomposable,
        'coverage': coverage,
        'component_vocab_size': vocab.component_vocab_size,
        'operator_vocab_size': vocab.operator_vocab_size,
        'depth_distribution': depth_dist,
        'max_depth': args.max_depth,
    }
    stats_path = args.output_dir / 'vocab_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()

def load_corpus_chars(corpus_dir: Path) -> Set[str]:
    """Load unique characters from corpus files."""
    chars: Set[str] = set()

    for jsonl_path in corpus_dir.glob('**/*.jsonl'):
        print(f"  Loading characters from {jsonl_path.name}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get('text', '')
                    for char in text:
                        cp = ord(char)
                        if (0x4E00 <= cp <= 0x9FFF or
                            0x3400 <= cp <= 0x4DBF or
                            0x20000 <= cp <= 0x2A6DF or
                            0x2F00 <= cp <= 0x2FDF or
                            0x2E80 <= cp <= 0x2EFF):
                            chars.add(char)
                except json.JSONDecodeError:
                    continue

    return chars


def load_bert_vocab_chars(vocab_file: Path) -> Set[str]:
    """Load CJK characters from BERT vocabulary file."""
    chars: Set[str] = set()

    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if len(token) == 1:
                cp = ord(token)
                if (0x4E00 <= cp <= 0x9FFF or
                    0x3400 <= cp <= 0x4DBF or
                    0x20000 <= cp <= 0x2A6DF):
                    chars.add(token)

    return chars


def main():
    parser = argparse.ArgumentParser(
        description='Prepare IDS data and build CNM vocabulary'
    )
    parser.add_argument(
        '--ids-file',
        type=Path,
        default=Path('data/ids/ids_parsed.json'),
        help='Path to parsed IDS JSON file'
    )
    parser.add_argument(
        '--corpus-dir',
        type=Path,
        help='Directory containing corpus JSONL files (optional)'
    )
    parser.add_argument(
        '--bert-vocab',
        type=Path,
        help='Path to BERT vocab.txt file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/ids'),
        help='Output directory for CNM vocabulary'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--min-component-freq',
        type=int,
        default=1,
        help='Minimum frequency for components in vocabulary'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load IDS data
    print(f"Loading IDS data from {args.ids_file}...")
    if not args.ids_file.exists():
        print(f"Error: IDS file not found: {args.ids_file}")
        print("Run scripts/download_ids.py first to download IDS data.")
        sys.exit(1)

    with open(args.ids_file, 'r', encoding='utf-8') as f:
        raw_ids = json.load(f)
        ids_data = normalize_ids_data(raw_ids)
    print(f"Loaded {len(ids_data)} characters with IDS decompositions (normalized)")


    # Create parser
    parser_instance = IDSParser(ids_data=ids_data, max_depth=args.max_depth)

    # Collect target characters
    target_chars: Set[str] = set()

    # From IDS data
    target_chars.update(ids_data.keys())
    print(f"Characters from IDS: {len(target_chars)}")

    # From corpus
    if args.corpus_dir and args.corpus_dir.exists():
        print(f"\nLoading characters from corpus: {args.corpus_dir}")
        corpus_chars = load_corpus_chars(args.corpus_dir)
        print(f"Characters from corpus: {len(corpus_chars)}")
        target_chars.update(corpus_chars)

    # From BERT vocab
    if args.bert_vocab and args.bert_vocab.exists():
        print(f"\nLoading characters from BERT vocab: {args.bert_vocab}")
        bert_chars = load_bert_vocab_chars(args.bert_vocab)
        print(f"Characters from BERT vocab: {len(bert_chars)}")
        target_chars.update(bert_chars)

    print(f"\nTotal target characters: {len(target_chars)}")

    # Parse all characters
    print("\nParsing characters...")
    parser_instance.parse_all(target_chars)
    for ch in ["你", "好", "我", "中", "国"]:
        try:
            tree = parser_instance.parse(ch) if hasattr(parser_instance, "parse") else None
            print(f"[DEBUG] {ch} parsed tree:", tree)
        except Exception as e:
            print(f"[DEBUG] {ch} parse error:", e)
    print(f"Parsed {parser_instance.num_cached} characters")

    # Compute coverage
    decomposable, total, coverage = estimate_ids_coverage(parser_instance, target_chars)
    print(f"\nIDS Coverage: {decomposable}/{total} = {coverage:.1%}")

    # Compute depth distribution
    depth_dist = parser_instance.compute_depth_distribution()
    print("\nTree depth distribution:")
    for depth, count in sorted(depth_dist.items()):
        print(f"  Depth {depth}: {count}")

    # Build vocabulary
    print("\nBuilding CNM vocabulary...")
    vocab = CNMVocab.build(
        parser_instance,
        target_chars,
        min_component_freq=args.min_component_freq,
    )

    print(f"\nVocabulary statistics:")
    print(f"  Components: {vocab.component_vocab_size}")
    print(f"  Operators: {vocab.operator_vocab_size}")
    print(f"  Characters with trees: {len(vocab.char_to_indexed)}")

    # Save vocabulary
    vocab_path = args.output_dir / 'cnm_vocab.json'
    vocab.save(vocab_path)
    print(f"\nSaved CNM vocabulary to {vocab_path}")

    # Save parser cache (for faster loading)
    cache_path = args.output_dir / 'ids_cache.json'
    parser_instance.save_cache(cache_path)
    print(f"Saved parse cache to {cache_path}")

    # Save statistics
    stats = {
        'total_characters': len(target_chars),
        'decomposable_characters': decomposable,
        'coverage': coverage,
        'component_vocab_size': vocab.component_vocab_size,
        'operator_vocab_size': vocab.operator_vocab_size,
        'depth_distribution': depth_dist,
        'max_depth': args.max_depth,
    }
    stats_path = args.output_dir / 'vocab_stats.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
