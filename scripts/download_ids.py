#!/usr/bin/env python3
"""
Download and parse BabelStone IDS (Ideographic Description Sequences) data.

The BabelStone IDS file contains decomposition information for CJK characters,
mapping each character to its component structure using IDC (Ideographic
Description Characters) operators.

Source: https://www.babelstone.co.uk/CJK/IDS.TXT
Format: U+XXXX <TAB> character <TAB> IDS1 [<TAB> IDS2 ...]

Example:
    U+660E	明	⿰日月	⿱日月
    (The character 明 can be decomposed as left-right 日月 or top-bottom 日月)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# BabelStone IDS data URL
BABELSTONE_IDS_URL = "https://www.babelstone.co.uk/CJK/IDS.TXT"

# Alternative: CHISE IDS repository (more structured, larger)
CHISE_IDS_URLS = [
    "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt",
]

# Ideographic Description Characters (IDCs)
# Binary operators: ⿰ ⿱ ⿴ ⿵ ⿶ ⿷ ⿸ ⿹ ⿺ ⿻
# Ternary operators: ⿲ ⿳
# New in Unicode 15.1: ⿾ ⿿
IDC_BINARY = set("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻")
IDC_TERNARY = set("⿲⿳")
IDC_NEW = set("⿾⿿")  # Subtraction and overlapping, treat as binary
IDC_ALL = IDC_BINARY | IDC_TERNARY | IDC_NEW

# Unicode Private Use Area ranges
PUA_RANGES = [
    (0xE000, 0xF8FF),    # BMP PUA
    (0xF0000, 0xFFFFD),  # Supplementary PUA-A
    (0x100000, 0x10FFFD), # Supplementary PUA-B
]


def is_pua(char: str) -> bool:
    """Check if a character is in a Private Use Area."""
    if len(char) != 1:
        return False
    cp = ord(char)
    return any(start <= cp <= end for start, end in PUA_RANGES)


def normalize_ids_string(ids_string: str) -> List[str]:
    """
    Normalize an IDS string by:
    1. Splitting alternatives separated by '^'
    2. Removing source annotations like $(GHTJKPV) or [GKT]
    3. Removing curly brace markers like {1} or {1+1}
    4. Removing non-IDC control characters like 〾 (U+303E)
    5. Stripping whitespace

    Args:
        ids_string: Raw IDS string from data source

    Returns:
        List of cleaned IDS alternatives
    """
    # First split by '^' to get alternatives
    alternatives = ids_string.split('^')

    cleaned: List[str] = []
    for alt in alternatives:
        alt = alt.strip()
        if not alt:
            continue

        # Remove source annotations: $(XYZ), $[XYZ], $(XYZ+ABC), etc.
        alt = re.sub(r'\$\([^)]*\)', '', alt)
        alt = re.sub(r'\$\[[^\]]*\]', '', alt)

        # Remove square bracket annotations: [GHTJKPV], [GK], etc.
        alt = re.sub(r'\[[A-Z0-9*+]+\]', '', alt)

        # Remove curly brace markers: {1}, {2}, {1+1}, etc.
        alt = re.sub(r'\{[^}]*\}', '', alt)

        # Remove ideographic variation indicator U+303E (〾)
        alt = alt.replace('\u303E', '')

        # Remove any other known noise characters
        # U+2FF2 and U+2FF3 are ternary IDCs, keep them
        # But remove U+3013 (〓 GETA MARK) if present
        alt = alt.replace('\u3013', '')

        # Remove stray asterisks used as placeholders
        alt = alt.replace('*', '')

        # Strip and check if anything remains
        alt = alt.strip()
        if alt and len(alt) > 0:
            # Validate: first char should be IDC or single char
            first_char = alt[0]
            if first_char in IDC_ALL or len(alt) == 1:
                cleaned.append(alt)

    return cleaned


def parse_ids_line(line: str) -> Optional[tuple[str, str, List[str]]]:
    """
    Parse a single line from the IDS file.

    Returns:
        Tuple of (codepoint, character, list of IDS sequences) or None if invalid.
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith("#") or line.startswith(";"):
        return None

    # Split by tabs
    parts = line.split("\t")
    if len(parts) < 3:
        return None

    codepoint = parts[0].strip()
    char = parts[1].strip()

    # Validate codepoint format
    if not re.match(r"U\+[0-9A-Fa-f]{4,5}", codepoint):
        return None

    # Validate character
    if len(char) != 1:
        return None

    # Extract all IDS sequences (may have multiple alternatives)
    # Join all IDS parts first, then normalize
    raw_ids_parts = []
    for part in parts[2:]:
        part = part.strip()
        if part:
            raw_ids_parts.append(part)

    # Normalize all raw parts, handling ^ separators and annotations
    ids_sequences: List[str] = []
    for raw_part in raw_ids_parts:
        normalized = normalize_ids_string(raw_part)
        for seq in normalized:
            if seq and seq not in ids_sequences:
                ids_sequences.append(seq)

    if not ids_sequences:
        return None

    return codepoint, char, ids_sequences


def download_babelstone_ids(output_path: Path, timeout: int = 60) -> Dict[str, List[str]]:
    """
    Download and parse BabelStone IDS data.

    Returns:
        Dictionary mapping characters to lists of IDS sequences.
    """
    print(f"Downloading IDS data from {BABELSTONE_IDS_URL}...")

    try:
        request = Request(
            BABELSTONE_IDS_URL,
            headers={"User-Agent": "CNM-IDS-Downloader/1.0"}
        )
        with urlopen(request, timeout=timeout) as response:
            content = response.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        print(f"Error downloading IDS data: {e}")
        sys.exit(1)

    # Save raw file
    raw_path = output_path / "IDS.TXT"
    raw_path.write_text(content, encoding="utf-8")
    print(f"Saved raw IDS data to {raw_path}")

    # Parse into structured format
    ids_data: Dict[str, List[str]] = {}
    pua_chars: List[str] = []

    for line in content.splitlines():
        result = parse_ids_line(line)
        if result is None:
            continue

        codepoint, char, sequences = result
        ids_data[char] = sequences

        # Track PUA characters for reference
        for seq in sequences:
            for c in seq:
                if is_pua(c) and c not in pua_chars:
                    pua_chars.append(c)

    print(f"Parsed {len(ids_data)} characters with IDS decompositions")
    print(f"Found {len(pua_chars)} unique PUA components")

    return ids_data


def download_chise_ids(output_path: Path, timeout: int = 60) -> Dict[str, List[str]]:
    """
    Download and parse CHISE IDS data (alternative source, often more complete).

    Returns:
        Dictionary mapping characters to lists of IDS sequences.
    """
    ids_data: Dict[str, List[str]] = {}

    for url in CHISE_IDS_URLS:
        print(f"Downloading IDS data from {url}...")
        try:
            request = Request(url, headers={"User-Agent": "CNM-IDS-Downloader/1.0"})
            with urlopen(request, timeout=timeout) as response:
                content = response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            print(f"Warning: Failed to download from {url}: {e}")
            continue

        # CHISE format is similar but may have slight differences
        for line in content.splitlines():
            result = parse_ids_line(line)
            if result is None:
                continue

            codepoint, char, sequences = result
            if char not in ids_data:
                ids_data[char] = sequences
            else:
                # Merge with existing sequences
                for seq in sequences:
                    if seq not in ids_data[char]:
                        ids_data[char].append(seq)

    print(f"Parsed {len(ids_data)} characters from CHISE sources")
    return ids_data


def compute_statistics(ids_data: Dict[str, List[str]]) -> Dict:
    """Compute statistics about the IDS data."""
    stats = {
        "total_characters": len(ids_data),
        "multi_decomposition": 0,
        "binary_ops": {op: 0 for op in IDC_BINARY},
        "ternary_ops": {op: 0 for op in IDC_TERNARY},
        "pua_components": set(),
        "max_depth": 0,
        "avg_sequences_per_char": 0,
    }

    total_sequences = 0

    for char, sequences in ids_data.items():
        total_sequences += len(sequences)
        if len(sequences) > 1:
            stats["multi_decomposition"] += 1

        for seq in sequences:
            for c in seq:
                if c in IDC_BINARY:
                    stats["binary_ops"][c] += 1
                elif c in IDC_TERNARY:
                    stats["ternary_ops"][c] += 1
                elif is_pua(c):
                    stats["pua_components"].add(c)

    stats["pua_components"] = len(stats["pua_components"])
    stats["avg_sequences_per_char"] = total_sequences / len(ids_data) if ids_data else 0

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and parse BabelStone/CHISE IDS data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ids"),
        help="Output directory for IDS data (default: data/ids)"
    )
    parser.add_argument(
        "--source",
        choices=["babelstone", "chise", "both"],
        default="babelstone",
        help="IDS data source (default: babelstone)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    ids_data: Dict[str, List[str]] = {}

    if args.source in ("babelstone", "both"):
        babelstone_data = download_babelstone_ids(args.output_dir, args.timeout)
        ids_data.update(babelstone_data)

    if args.source in ("chise", "both"):
        chise_data = download_chise_ids(args.output_dir, args.timeout)
        # Merge CHISE data (CHISE may have additional characters)
        for char, sequences in chise_data.items():
            if char not in ids_data:
                ids_data[char] = sequences
            else:
                for seq in sequences:
                    if seq not in ids_data[char]:
                        ids_data[char].append(seq)

    # Compute and print statistics
    stats = compute_statistics(ids_data)
    print("\n=== IDS Data Statistics ===")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Characters with multiple decompositions: {stats['multi_decomposition']}")
    print(f"Unique PUA components: {stats['pua_components']}")
    print(f"Avg sequences per character: {stats['avg_sequences_per_char']:.2f}")
    print("\nBinary operator usage:")
    for op, count in sorted(stats["binary_ops"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {op}: {count}")
    print("Ternary operator usage:")
    for op, count in sorted(stats["ternary_ops"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {op}: {count}")

    # Save parsed data as JSON
    json_path = args.output_dir / "ids_parsed.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ids_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved parsed IDS data to {json_path}")

    # Save statistics
    stats_path = args.output_dir / "ids_stats.json"
    # Convert sets to lists for JSON serialization
    stats_serializable = {
        k: (list(v) if isinstance(v, set) else v)
        for k, v in stats.items()
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
