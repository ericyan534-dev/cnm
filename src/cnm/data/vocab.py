"""
Vocabulary management for CNM structural embeddings.

This module manages vocabularies for:
- Components: atomic radicals and leaf characters
- Operators: IDC (Ideographic Description Characters)

The vocabularies map characters/operators to integer indices for use
in embedding layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from cnm.data.tree import IDSTree, IndexedTree, BatchedTrees, IDC_ALL
from cnm.data.ids_parser import IDSParser, is_pua


# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
FALLBACK_TOKEN = "[FALLBACK]"  # For non-decomposable tokens


@dataclass
class CNMVocab:
    """
    Vocabulary for CNM structural embeddings.

    Manages two vocabularies:
    1. Components: Atomic elements (radicals, PUA chars, etc.)
    2. Operators: IDC characters

    Attributes:
        component_to_id: Mapping from component chars to indices
        operator_to_id: Mapping from IDC operators to indices
        id_to_component: Reverse mapping for components
        id_to_operator: Reverse mapping for operators
        char_to_tree: Mapping from characters to parsed IDSTree
    """
    component_to_id: Dict[str, int] = field(default_factory=dict)
    operator_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_component: Dict[int, str] = field(default_factory=dict)
    id_to_operator: Dict[int, str] = field(default_factory=dict)
    char_to_tree: Dict[str, IDSTree] = field(default_factory=dict)
    char_to_indexed: Dict[str, IndexedTree] = field(default_factory=dict)

    # Standard IDC operators
    IDC_OPERATORS = list(IDC_ALL)

    def __post_init__(self):
        """Initialize special tokens if vocabularies are empty."""
        if not self.component_to_id:
            self._init_component_vocab()
        if not self.operator_to_id:
            self._init_operator_vocab()

    def _init_component_vocab(self):
        """Initialize component vocabulary with special tokens."""
        self.component_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            FALLBACK_TOKEN: 2,
        }
        self.id_to_component = {v: k for k, v in self.component_to_id.items()}

    def _init_operator_vocab(self):
        """Initialize operator vocabulary with special tokens and all IDCs."""
        self.operator_to_id = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
        }
        # Add all IDC operators
        for i, op in enumerate(self.IDC_OPERATORS, start=2):
            self.operator_to_id[op] = i

        self.id_to_operator = {v: k for k, v in self.operator_to_id.items()}

    @classmethod
    def build(
        cls,
        parser: IDSParser,
        corpus_chars: Set[str],
        min_component_freq: int = 1,
    ) -> "CNMVocab":
        """
        Build vocabulary from a corpus of characters.

        This method:
        1. Parses all characters in the corpus
        2. Collects all unique components and operators
        3. Builds vocabulary mappings
        4. Caches indexed trees for efficiency

        Args:
            parser: IDSParser instance with loaded IDS data
            corpus_chars: Set of characters to include
            min_component_freq: Minimum frequency for a component to be included

        Returns:
            CNMVocab with populated vocabularies
        """
        vocab = cls()

        # Parse all characters
        component_counts: Dict[str, int] = {}
        for char in corpus_chars:
            tree = parser.parse(char)
            vocab.char_to_tree[char] = tree

            # Count components
            for leaf in tree.leaves():
                component_counts[leaf.char] = component_counts.get(leaf.char, 0) + 1

        # Build component vocabulary (sorted by frequency)
        sorted_components = sorted(
            component_counts.items(),
            key=lambda x: (-x[1], x[0])  # Sort by freq desc, then alphabetically
        )

        next_id = len(vocab.component_to_id)
        for component, count in sorted_components:
            if count >= min_component_freq and component not in vocab.component_to_id:
                vocab.component_to_id[component] = next_id
                vocab.id_to_component[next_id] = component
                next_id += 1

        # Build indexed trees
        for char, tree in vocab.char_to_tree.items():
            indexed = tree.to_indexed(
                vocab.component_to_id,
                vocab.operator_to_id,
                unk_component_id=vocab.unk_component_id,
                unk_operator_id=vocab.unk_operator_id,
            )
            vocab.char_to_indexed[char] = indexed

        return vocab

    @property
    def component_vocab_size(self) -> int:
        """Size of component vocabulary."""
        return len(self.component_to_id)

    @property
    def operator_vocab_size(self) -> int:
        """Size of operator vocabulary."""
        return len(self.operator_to_id)

    @property
    def pad_component_id(self) -> int:
        """ID of PAD token in component vocab."""
        return self.component_to_id[PAD_TOKEN]

    @property
    def unk_component_id(self) -> int:
        """ID of UNK token in component vocab."""
        return self.component_to_id[UNK_TOKEN]

    @property
    def fallback_component_id(self) -> int:
        """ID of FALLBACK token in component vocab."""
        return self.component_to_id[FALLBACK_TOKEN]

    @property
    def pad_operator_id(self) -> int:
        """ID of PAD token in operator vocab."""
        return self.operator_to_id[PAD_TOKEN]

    @property
    def unk_operator_id(self) -> int:
        """ID of UNK token in operator vocab."""
        return self.operator_to_id[UNK_TOKEN]

    def get_component_id(self, component: str) -> int:
        """Get ID for a component, defaulting to UNK."""
        return self.component_to_id.get(component, self.unk_component_id)

    def get_operator_id(self, operator: str) -> int:
        """Get ID for an operator, defaulting to UNK."""
        return self.operator_to_id.get(operator, self.unk_operator_id)

    def get_tree(self, char: str) -> Optional[IDSTree]:
        """Get parsed tree for a character."""
        return self.char_to_tree.get(char)

    def get_indexed_tree(self, char: str) -> Optional[IndexedTree]:
        """Get indexed tree for a character."""
        return self.char_to_indexed.get(char)

    def batch_trees(
        self,
        chars: List[str],
        device: Optional[torch.device] = None,
    ) -> BatchedTrees:
        """
        Batch multiple characters' trees into a single BatchedTrees.

        For characters not in the vocabulary, creates a simple leaf tree
        with the FALLBACK component.

        Args:
            chars: List of characters to batch
            device: Device to place tensors on

        Returns:
            BatchedTrees containing all characters
        """
        trees: List[IndexedTree] = []

        for char in chars:
            indexed = self.char_to_indexed.get(char)
            if indexed is not None:
                trees.append(indexed)
            else:
                # Create fallback leaf for unknown characters
                trees.append(IndexedTree(
                    component_id=self.fallback_component_id,
                    operator_id=0,
                    children=(),
                    depth=0,
                ))

        return BatchedTrees.from_trees(trees, device=device)

    def get_unique_chars(self, struct_ids: torch.LongTensor) -> Tuple[List[str], torch.LongTensor]:
        """
        Get unique characters from struct_ids tensor and their mapping.

        Args:
            struct_ids: [batch, seq] tensor of character indices

        Returns:
            Tuple of (list of unique chars, [batch, seq] mapping to unique indices)
        """
        flat = struct_ids.flatten()
        unique_ids = torch.unique(flat)

        # Map IDs back to characters
        unique_chars = []
        id_to_unique_idx = {}

        for i, uid in enumerate(unique_ids.tolist()):
            # Lookup character from ID (if we stored it)
            # For now, just use the ID as-is
            id_to_unique_idx[uid] = i
            unique_chars.append(str(uid))  # Placeholder

        # Create mapping tensor
        mapping = torch.tensor(
            [id_to_unique_idx[i.item()] for i in flat],
            dtype=torch.long,
            device=struct_ids.device,
        ).view_as(struct_ids)

        return unique_chars, mapping

    def add_component(self, component: str) -> int:
        """Add a new component to the vocabulary."""
        if component in self.component_to_id:
            return self.component_to_id[component]

        new_id = len(self.component_to_id)
        self.component_to_id[component] = new_id
        self.id_to_component[new_id] = component
        return new_id

    def save(self, path: Path) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            "component_to_id": self.component_to_id,
            "operator_to_id": self.operator_to_id,
            "char_to_tree": {
                char: _serialize_indexed_tree(tree)
                for char, tree in self.char_to_indexed.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CNMVocab":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.component_to_id = data["component_to_id"]
        vocab.operator_to_id = data["operator_to_id"]
        vocab.id_to_component = {int(v): k for k, v in vocab.component_to_id.items()}
        vocab.id_to_operator = {int(v): k for k, v in vocab.operator_to_id.items()}

        # Reconstruct indexed trees
        for char, tree_data in data.get("char_to_tree", {}).items():
            vocab.char_to_indexed[char] = _deserialize_indexed_tree(tree_data)

        return vocab

    def get_component_labels(self, char: str) -> torch.Tensor:
        """
        Get multi-hot component labels for a character.

        Used for the auxiliary component prediction task.

        Args:
            char: The character to get labels for

        Returns:
            [component_vocab_size] binary tensor
        """
        labels = torch.zeros(self.component_vocab_size, dtype=torch.float)

        tree = self.char_to_tree.get(char)
        if tree is None:
            return labels

        for leaf in tree.leaves():
            comp_id = self.get_component_id(leaf.char)
            if comp_id != self.unk_component_id:
                labels[comp_id] = 1.0

        return labels


def _serialize_indexed_tree(tree: IndexedTree) -> Dict:
    """Serialize IndexedTree to dictionary."""
    result = {
        "component_id": tree.component_id,
        "operator_id": tree.operator_id,
        "depth": tree.depth,
    }
    if tree.children:
        result["children"] = [_serialize_indexed_tree(c) for c in tree.children]
    return result


def _deserialize_indexed_tree(data: Dict) -> IndexedTree:
    """Deserialize IndexedTree from dictionary."""
    children = tuple(
        _deserialize_indexed_tree(c)
        for c in data.get("children", [])
    )
    return IndexedTree(
        component_id=data["component_id"],
        operator_id=data["operator_id"],
        children=children,
        depth=data["depth"],
    )


def extract_corpus_chars(texts: List[str]) -> Set[str]:
    """
    Extract all unique CJK characters from a corpus.

    Args:
        texts: List of text strings

    Returns:
        Set of unique CJK characters
    """
    chars: Set[str] = set()
    for text in texts:
        for char in text:
            # Only include CJK characters
            cp = ord(char)
            if (0x4E00 <= cp <= 0x9FFF or      # CJK Unified
                0x3400 <= cp <= 0x4DBF or      # Extension A
                0x20000 <= cp <= 0x2A6DF or    # Extension B
                0xF900 <= cp <= 0xFAFF or      # Compatibility
                0x2F00 <= cp <= 0x2FDF or      # Kangxi Radicals
                0x2E80 <= cp <= 0x2EFF):       # Radicals Supplement
                chars.add(char)
    return chars
