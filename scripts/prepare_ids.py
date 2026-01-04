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

def normalize_ids_data(raw) -> dict:
    """
    Normalize various IDS JSON formats into:
        { "<char>": "<ids_string>" }

    Supported inputs:
      - dict: {char: "⿰亻尔"}
      - dict: {char: {"ids": "⿰亻尔", ...}}
      - dict: {char: ["⿰","亻","尔"]}  (tokens list)
      - list: [{"char": "你", "ids": "⿰亻尔"}, ...]
      - list: [{"character": "你", "ids": "⿰亻尔"}, ...]
    """
    if isinstance(raw, dict):
        # Detect value type
        if not raw:
            return {}
        v0 = next(iter(raw.values()))
        if isinstance(v0, str):
            return raw
        if isinstance(v0, dict):
            # Common: {char: {"ids": "..."}}
            out = {}
            for ch, obj in raw.items():
                if isinstance(obj, dict) and "ids" in obj and isinstance(obj["ids"], str):
                    out[ch] = obj["ids"]
            return out
        if isinstance(v0, list):
            # Common: {char: ["⿰","亻","尔"]}
            out = {}
            for ch, toks in raw.items():
                if isinstance(toks, list) and all(isinstance(t, str) for t in toks):
                    out[ch] = "".join(toks)
            return out
        raise TypeError(f"Unsupported dict value type: {type(v0)}")

    if isinstance(raw, list):
        out = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            ch = item.get("char") or item.get("character") or item.get("ch")
            ids = item.get("ids") or item.get("decomp") or item.get("decomposition")
            if isinstance(ch, str) and len(ch) == 1 and isinstance(ids, str) and ids:
                out[ch] = ids
        return out

    raise TypeError(f"Unsupported IDS JSON top-level type: {type(raw)}")


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

import unicodedata
import re

_IDS_WS = re.compile(r"\s+")

def normalize_ids_mapping(raw) -> dict[str, str]:
    """
    Convert many IDS JSON formats into:
        { "<single_char>": "<ids_string_without_spaces>" }

    Handles keys:
      - actual char keys: "你"
      - integer codepoints: 20320
      - hex strings: "4F60"
      - "U+4F60" / "u+4f60"
    Handles values:
      - string IDS: "⿰亻尔"
      - dict with ids field: {"ids": "⿰亻尔", ...}
      - list tokens: ["⿰", "亻", "尔"]
    """
    def key_to_char(k) -> str | None:
        # Already a single char
        if isinstance(k, str):
            ks = k.strip()
            if len(ks) == 1:
                return ks
            # U+XXXX
            if ks.upper().startswith("U+"):
                try:
                    return chr(int(ks[2:], 16))
                except Exception:
                    return None
            # plain hex like "4F60"
            if all(c in "0123456789abcdefABCDEF" for c in ks) and 2 <= len(ks) <= 6:
                try:
                    return chr(int(ks, 16))
                except Exception:
                    return None
            return None

        # int codepoint
        if isinstance(k, int):
            try:
                return chr(k)
            except Exception:
                return None

        return None

    def val_to_ids(v) -> str | None:
        if isinstance(v, str):
            s = v
        elif isinstance(v, dict):
            # common field names
            for field in ("ids", "decomp", "decomposition", "ids_str", "value"):
                if field in v and isinstance(v[field], str):
                    s = v[field]
                    break
            else:
                return None
        elif isinstance(v, list) and all(isinstance(t, str) for t in v):
            s = "".join(v)
        else:
            return None

        # Normalize + strip whitespace
        s = unicodedata.normalize("NFKC", s)
        s = _IDS_WS.sub("", s)

        s = sanitize_ids_string(s)
        return s if s else None


    out: dict[str, str] = {}

    if isinstance(raw, dict):
        for k, v in raw.items():
            ch = key_to_char(k)
            ids = val_to_ids(v)
            if ch is not None and ids is not None:
                out[ch] = ids
        return out

    if isinstance(raw, list):
        # list of records: {"char": "...", "ids": "..."}
        for item in raw:
            if not isinstance(item, dict):
                continue
            ch = item.get("char") or item.get("character") or item.get("ch")
            ids = item.get("ids") or item.get("decomp") or item.get("decomposition")
            if isinstance(ch, str) and len(ch.strip()) == 1 and isinstance(ids, str):
                s = unicodedata.normalize("NFKC", ids)
                s = _IDS_WS.sub("", s)
                out[ch.strip()] = s
        return out

    raise TypeError(f"Unsupported IDS JSON top-level type: {type(raw)}")
import re
import unicodedata

# strips $(...) metadata and {..} annotations
_RE_META = re.compile(r"\$\([^)]*\)")
_RE_BRACE = re.compile(r"\{[^}]*\}")

def sanitize_ids_string(raw: str) -> str:
    """
    Turn non-standard IDS strings like:
      '^⿰亻尔$(GHTJKPV)^⿰亻尓$(Z)'
    into a clean IDS string like:
      '⿰亻尔'
    Strategy:
      - NFKC normalize
      - remove $(...) and {...}
      - split alternative parses by '^' and pick the best candidate
      - drop remaining '^'
      - drop ASCII junk (metadata tags)
    """
    if not raw:
        return ""

    s = unicodedata.normalize("NFKC", raw)

    # remove obvious metadata/annotations
    s = _RE_META.sub("", s)
    s = _RE_BRACE.sub("", s)

    # split by caret (this file uses ^ as variant separator AND also prefixes)
    parts = [p for p in s.split("^") if p]

    # choose the best candidate: prefer ones that contain IDS operators (⿰⿱... U+2FF0..U+2FFB),
    # then by length (more informative)
    def score(p: str) -> tuple[int, int]:
        op_cnt = sum(0x2FF0 <= ord(c) <= 0x2FFB for c in p)
        return (op_cnt, len(p))

    cand = max(parts, key=score) if parts else s

    # drop any remaining carets
    cand = cand.replace("^", "")

    # drop ASCII (your metadata tags like GHTJKPV, Z, X are ASCII)
    cand = "".join(ch for ch in cand if ord(ch) >= 128)

    return cand.strip()

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

    ids_data = normalize_ids_mapping(raw_ids)
    print(f"Loaded {len(ids_data)} characters with IDS decompositions (normalized)")

    def fail_fast_ids_mapping(ids_data: dict[str, str]) -> None:
    # 1) Ensure common characters exist and look sane
        must = ["你", "好", "我", "中", "国"]
        missing = [c for c in must if c not in ids_data]
        if missing:
            raise RuntimeError(f"IDS mapping missing basic chars: {missing}. Wrong IDS file or key normalization.")

        for c in must:
            v = ids_data[c]
            if "^" in v or "$(" in v or "{" in v:
                raise RuntimeError(f"IDS for {c} still contains metadata after sanitization: {repr(v)}")

    # 2) Operator presence ratio (quick proxy for parseability)
        vals = list(ids_data.values())
        sample = vals[:5000] if len(vals) > 5000 else vals
        op_frac = sum(any(0x2FF0 <= ord(ch) <= 0x2FFB for ch in s) for s in sample) / max(1, len(sample))

    # If this is near 0, your IDS strings still aren't standard IDS
        if op_frac < 0.05:
            raise RuntimeError(f"IDS operator presence too low ({op_frac:.3f}). IDS strings likely still malformed.")

        fail_fast_ids_mapping(ids_data)
        print("[OK] IDS mapping passed fail-fast checks.")
    def fail_fast_parser(parser_instance: IDSParser) -> None:
        tests = ["你", "好", "我", "中", "国"]
        bad = []
        for ch in tests:
            try:
                t = parser_instance.parse(ch) if hasattr(parser_instance, "parse") else None
            # These fields match your earlier debug output
                if (t is None) or (getattr(t, "depth", 0) == 0) or (getattr(t, "operator", None) is None):
                    bad.append((ch, t))
            except Exception as e:
                bad.append((ch, f"EXC: {e}"))

        if bad:
            msg = "\n".join([f"  {c}: {repr(t)[:160]}" for c, t in bad])
            raise RuntimeError("IDSParser sanity check failed (still producing leaf trees). Examples:\n" + msg)

    fail_fast_parser(parser_instance)
    print("[OK] IDSParser passed fail-fast checks.")





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


