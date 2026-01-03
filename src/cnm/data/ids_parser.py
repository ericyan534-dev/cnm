"""
IDS (Ideographic Description Sequences) parser for Chinese characters.

This module parses IDS strings into tree structures, handling:
- Multiple decomposition alternatives (chooses shortest)
- Circular dependencies (detects cycles, returns atomic)
- Maximum depth limits (truncates to leaf)
- PUA (Private Use Area) characters
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from cnm.data.tree import IDSTree, IDC_ARITY, IDC_ALL


# Unicode Private Use Area ranges
PUA_RANGES = [
    (0xE000, 0xF8FF),     # BMP PUA
    (0xF0000, 0xFFFFD),   # Supplementary PUA-A
    (0x100000, 0x10FFFD), # Supplementary PUA-B
]


def is_pua(char: str) -> bool:
    """Check if a character is in a Private Use Area."""
    if len(char) != 1:
        return False
    cp = ord(char)
    return any(start <= cp <= end for start, end in PUA_RANGES)


def is_cjk(char: str) -> bool:
    """Check if a character is in CJK Unicode blocks."""
    if len(char) != 1:
        return False
    cp = ord(char)
    # CJK Unified Ideographs and extensions
    return (
        0x4E00 <= cp <= 0x9FFF or      # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or      # CJK Extension A
        0x20000 <= cp <= 0x2A6DF or    # CJK Extension B
        0x2A700 <= cp <= 0x2B73F or    # CJK Extension C
        0x2B740 <= cp <= 0x2B81F or    # CJK Extension D
        0x2B820 <= cp <= 0x2CEAF or    # CJK Extension E
        0x2CEB0 <= cp <= 0x2EBEF or    # CJK Extension F
        0x30000 <= cp <= 0x3134F or    # CJK Extension G
        0x31350 <= cp <= 0x323AF or    # CJK Extension H
        0x2EBF0 <= cp <= 0x2EE5F or    # CJK Extension I
        0xF900 <= cp <= 0xFAFF or      # CJK Compatibility Ideographs
        0x2F00 <= cp <= 0x2FDF or      # Kangxi Radicals
        0x2E80 <= cp <= 0x2EFF         # CJK Radicals Supplement
    )


@dataclass
class ParseResult:
    """Result of parsing an IDS string."""
    tree: Optional[IDSTree]
    depth: int
    success: bool
    error: Optional[str] = None


class IDSParser:
    """
    Parser for converting IDS strings into IDSTree structures.

    The parser handles:
    - Multiple decomposition alternatives: selects the shortest tree
    - Circular dependencies: detects cycles and returns atomic leaf
    - Maximum depth limits: truncates to leaf node if exceeded
    - PUA characters: marks them with is_pua flag

    Attributes:
        raw_ids: Mapping from characters to lists of IDS strings
        max_depth: Maximum allowed tree depth before truncation
        cache: Cache of parsed trees for efficiency
    """

    def __init__(
        self,
        ids_data: Optional[Dict[str, List[str]]] = None,
        ids_path: Optional[Path] = None,
        max_depth: int = 6,
    ):
        """
        Initialize the parser.

        Args:
            ids_data: Pre-loaded dictionary of char -> IDS sequences
            ids_path: Path to JSON file with IDS data
            max_depth: Maximum tree depth before truncation (default: 6)
        """
        self.max_depth = max_depth
        self._cache: Dict[str, IDSTree] = {}
        self._in_progress: Set[str] = set()  # For cycle detection

        if ids_data is not None:
            self.raw_ids = ids_data
        elif ids_path is not None:
            with open(ids_path, "r", encoding="utf-8") as f:
                self.raw_ids = json.load(f)
        else:
            self.raw_ids = {}

    def parse(self, char: str, current_depth: int = 0) -> IDSTree:
        """
        Parse a character into an IDSTree.

        This is the main entry point. It handles caching, cycle detection,
        and selects the shortest decomposition among alternatives.

        Args:
            char: The character to parse
            current_depth: Current recursion depth (for max depth check)

        Returns:
            IDSTree representing the character's structure
        """
        # Check cache first
        if char in self._cache:
            cached = self._cache[char]
            # Adjust depth if we're deeper than cached tree's context
            if current_depth > 0 and current_depth + cached.depth > self.max_depth:
                return IDSTree.leaf(char, is_pua=is_pua(char))
            return cached

        # Cycle detection
        if char in self._in_progress:
            return IDSTree.leaf(char, is_pua=is_pua(char))

        # Max depth check
        if current_depth >= self.max_depth:
            return IDSTree.leaf(char, is_pua=is_pua(char))

        # If no IDS data for this char, return as atomic
        if char not in self.raw_ids:
            tree = IDSTree.leaf(char, is_pua=is_pua(char))
            self._cache[char] = tree
            return tree

        # Mark as in-progress for cycle detection
        self._in_progress.add(char)

        try:
            # Try all decompositions, select the shortest
            best_tree: Optional[IDSTree] = None
            best_depth = float("inf")

            for ids_string in self.raw_ids[char]:
                result = self._parse_ids_string(ids_string, char, current_depth)
                if result.success and result.tree is not None:
                    if result.depth < best_depth:
                        best_tree = result.tree
                        best_depth = result.depth

            if best_tree is not None:
                self._cache[char] = best_tree
                return best_tree

            # Fallback to atomic if no valid decomposition
            tree = IDSTree.leaf(char, is_pua=is_pua(char))
            self._cache[char] = tree
            return tree

        finally:
            self._in_progress.discard(char)

    def _parse_ids_string(
        self, ids_string: str, root_char: str, current_depth: int
    ) -> ParseResult:
        """
        Parse a single IDS string into a tree.

        IDS strings have the format: <operator><component1><component2>[<component3>]
        where components can themselves be IDS strings (nested structure).

        Args:
            ids_string: The IDS string to parse
            root_char: The character this tree represents
            current_depth: Current recursion depth

        Returns:
            ParseResult with the parsed tree and metadata
        """
        if not ids_string:
            return ParseResult(tree=None, depth=0, success=False, error="Empty IDS string")

        try:
            tree, consumed = self._parse_expression(ids_string, 0, root_char, current_depth)
            if tree is None:
                return ParseResult(tree=None, depth=0, success=False, error="Parse failed")
            if consumed != len(ids_string):
                return ParseResult(
                    tree=None, depth=0, success=False,
                    error=f"Incomplete parse: consumed {consumed}/{len(ids_string)}"
                )
            return ParseResult(tree=tree, depth=tree.depth, success=True)
        except Exception as e:
            return ParseResult(tree=None, depth=0, success=False, error=str(e))

    def _parse_expression(
        self,
        ids_string: str,
        pos: int,
        root_char: str,
        current_depth: int,
    ) -> Tuple[Optional[IDSTree], int]:
        """
        Parse an IDS expression starting at position pos.

        Returns:
            Tuple of (parsed tree, number of characters consumed)
        """
        if pos >= len(ids_string):
            return None, pos

        char = ids_string[pos]

        # If it's an IDC operator, parse its arguments
        if char in IDC_ALL:
            arity = IDC_ARITY.get(char, 2)
            children: List[IDSTree] = []
            next_pos = pos + 1

            for _ in range(arity):
                if next_pos >= len(ids_string):
                    return None, pos  # Not enough arguments
                child, next_pos = self._parse_expression(
                    ids_string, next_pos, root_char, current_depth + 1
                )
                if child is None:
                    return None, pos
                children.append(child)

            tree = IDSTree.internal(root_char, char, tuple(children))
            return tree, next_pos

        # Otherwise, it's a component character
        # If the component has its own IDS decomposition, recursively parse
        if char in self.raw_ids and current_depth < self.max_depth:
            subtree = self.parse(char, current_depth + 1)
            return subtree, pos + 1

        # Atomic component
        return IDSTree.leaf(char, is_pua=is_pua(char)), pos + 1

    def parse_all(self, chars: Set[str]) -> Dict[str, IDSTree]:
        """
        Parse all characters in a set.

        Args:
            chars: Set of characters to parse

        Returns:
            Dictionary mapping characters to their parsed trees
        """
        result = {}
        for char in chars:
            result[char] = self.parse(char)
        return result

    def get_all_components(self) -> Set[str]:
        """
        Get all unique atomic components across all parsed trees.

        Returns:
            Set of component characters
        """
        components: Set[str] = set()
        for tree in self._cache.values():
            for leaf in tree.leaves():
                components.add(leaf.char)
        return components

    def get_all_operators(self) -> Set[str]:
        """
        Get all unique operators used across all parsed trees.

        Returns:
            Set of IDC operators
        """
        operators: Set[str] = set()
        for tree in self._cache.values():
            for op in tree.operators():
                operators.add(op)
        return operators

    def compute_depth_distribution(self) -> Dict[int, int]:
        """
        Compute distribution of tree depths in the cache.

        Returns:
            Dictionary mapping depth to count
        """
        distribution: Dict[int, int] = {}
        for tree in self._cache.values():
            depth = tree.depth
            distribution[depth] = distribution.get(depth, 0) + 1
        return distribution

    def clear_cache(self) -> None:
        """Clear the parse cache."""
        self._cache.clear()

    @property
    def num_cached(self) -> int:
        """Number of characters in cache."""
        return len(self._cache)

    def save_cache(self, path: Path) -> None:
        """
        Save the parse cache to a file.

        The cache is saved as a JSON file with serialized tree structures.
        """
        serialized = {}
        for char, tree in self._cache.items():
            serialized[char] = _serialize_tree(tree)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

    def load_cache(self, path: Path) -> None:
        """Load a previously saved parse cache."""
        with open(path, "r", encoding="utf-8") as f:
            serialized = json.load(f)

        for char, data in serialized.items():
            self._cache[char] = _deserialize_tree(data)


def _serialize_tree(tree: IDSTree) -> Dict:
    """Serialize an IDSTree to a JSON-compatible dictionary."""
    result = {
        "char": tree.char,
        "depth": tree.depth,
        "is_pua": tree.is_pua,
    }
    if tree.operator is not None:
        result["operator"] = tree.operator
        result["children"] = [_serialize_tree(child) for child in tree.children]
    return result


def _deserialize_tree(data: Dict) -> IDSTree:
    """Deserialize an IDSTree from a dictionary."""
    if "operator" not in data:
        return IDSTree.leaf(data["char"], is_pua=data.get("is_pua", False))

    children = tuple(_deserialize_tree(child) for child in data["children"])
    return IDSTree(
        char=data["char"],
        operator=data["operator"],
        children=children,
        depth=data["depth"],
        is_pua=data.get("is_pua", False),
    )


def estimate_ids_coverage(
    parser: IDSParser, chars: Set[str]
) -> Tuple[int, int, float]:
    """
    Estimate IDS coverage for a set of characters.

    Args:
        parser: The parser to use
        chars: Set of characters to check

    Returns:
        Tuple of (decomposable_count, total_count, coverage_ratio)
    """
    decomposable = 0
    total = len(chars)

    for char in chars:
        tree = parser.parse(char)
        if not tree.is_leaf:  # Has decomposition
            decomposable += 1

    coverage = decomposable / total if total > 0 else 0.0
    return decomposable, total, coverage
