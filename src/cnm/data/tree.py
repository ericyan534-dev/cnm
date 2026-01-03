"""
Tree data structures for representing Chinese character decompositions.

This module defines the core data structures used to represent IDS
(Ideographic Description Sequences) trees and their batched forms
for efficient neural network processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import torch


# Ideographic Description Characters (IDCs)
IDC_BINARY = frozenset("⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻⿾⿿")
IDC_TERNARY = frozenset("⿲⿳")
IDC_ALL = IDC_BINARY | IDC_TERNARY

# Arity of each IDC operator
IDC_ARITY: Dict[str, int] = {
    "⿰": 2,  # Left to right
    "⿱": 2,  # Above to below
    "⿲": 3,  # Left to middle to right
    "⿳": 3,  # Above to middle to below
    "⿴": 2,  # Full surround
    "⿵": 2,  # Surround from above
    "⿶": 2,  # Surround from below
    "⿷": 2,  # Surround from left
    "⿸": 2,  # Surround from upper left
    "⿹": 2,  # Surround from upper right
    "⿺": 2,  # Surround from lower left
    "⿻": 2,  # Overlaid
    "⿾": 2,  # Subtraction (Unicode 15.1)
    "⿿": 2,  # Horizontal reflection (Unicode 15.1)
}


@dataclass(frozen=True)
class IDSTree:
    """
    Immutable tree representation of a Chinese character's IDS decomposition.

    The tree is structured as follows:
    - Leaf nodes represent atomic components (radicals or undecomposable characters)
    - Internal nodes have an IDC operator and 2-3 children (depending on operator arity)

    Attributes:
        char: The character this tree represents (for debugging/reference)
        operator: The IDC operator at this node, or None for leaf nodes
        children: Tuple of child trees (empty for leaves, 2-3 for internal nodes)
        depth: The depth of this subtree (leaves have depth 0)
        is_pua: Whether this character is in a Private Use Area
    """
    char: str
    operator: Optional[str] = None
    children: Tuple["IDSTree", ...] = ()
    depth: int = 0
    is_pua: bool = False

    def __post_init__(self) -> None:
        """Validate tree structure."""
        if self.operator is not None:
            expected_arity = IDC_ARITY.get(self.operator, 2)
            if len(self.children) != expected_arity:
                raise ValueError(
                    f"Operator {self.operator} expects {expected_arity} children, "
                    f"got {len(self.children)}"
                )
        elif self.children:
            raise ValueError("Leaf nodes should not have children")

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (atomic component)."""
        return self.operator is None

    @property
    def arity(self) -> int:
        """Get the arity of the operator (0 for leaves)."""
        if self.operator is None:
            return 0
        return IDC_ARITY.get(self.operator, 2)

    def num_nodes(self) -> int:
        """Count total number of nodes in the tree."""
        if self.is_leaf:
            return 1
        return 1 + sum(child.num_nodes() for child in self.children)

    def num_leaves(self) -> int:
        """Count number of leaf nodes (components)."""
        if self.is_leaf:
            return 1
        return sum(child.num_leaves() for child in self.children)

    def leaves(self) -> Iterator[IDSTree]:
        """Iterate over all leaf nodes in left-to-right order."""
        if self.is_leaf:
            yield self
        else:
            for child in self.children:
                yield from child.leaves()

    def operators(self) -> Iterator[str]:
        """Iterate over all operators in the tree (pre-order)."""
        if self.operator is not None:
            yield self.operator
            for child in self.children:
                yield from child.operators()

    def to_ids_string(self) -> str:
        """Convert tree back to IDS string representation."""
        if self.is_leaf:
            return self.char
        return self.operator + "".join(child.to_ids_string() for child in self.children)

    def to_indexed(
        self,
        component_to_id: Dict[str, int],
        operator_to_id: Dict[str, int],
        unk_component_id: int = 1,
        unk_operator_id: int = 1,
    ) -> "IndexedTree":
        """
        Convert to IndexedTree with integer indices.

        Args:
            component_to_id: Mapping from component chars to indices
            operator_to_id: Mapping from operators to indices
            unk_component_id: ID for unknown components (default: 1)
            unk_operator_id: ID for unknown operators (default: 1)

        Returns:
            IndexedTree with the same structure but integer indices
        """
        if self.is_leaf:
            comp_id = component_to_id.get(self.char, unk_component_id)
            return IndexedTree(
                component_id=comp_id,
                operator_id=0,  # No operator for leaves
                children=(),
                depth=self.depth,
            )

        op_id = operator_to_id.get(self.operator, unk_operator_id)
        indexed_children = tuple(
            child.to_indexed(component_to_id, operator_to_id, unk_component_id, unk_operator_id)
            for child in self.children
        )
        return IndexedTree(
            component_id=0,  # Internal nodes don't have components
            operator_id=op_id,
            children=indexed_children,
            depth=self.depth,
        )

    @classmethod
    def leaf(cls, char: str, is_pua: bool = False) -> "IDSTree":
        """Create a leaf node (atomic component)."""
        return cls(char=char, operator=None, children=(), depth=0, is_pua=is_pua)

    @classmethod
    def internal(cls, char: str, operator: str, children: Tuple["IDSTree", ...]) -> "IDSTree":
        """Create an internal node with operator and children."""
        child_depth = max(child.depth for child in children) if children else 0
        return cls(
            char=char,
            operator=operator,
            children=children,
            depth=child_depth + 1,
        )


@dataclass(frozen=True)
class IndexedTree:
    """
    Tree with integer indices instead of string characters/operators.

    This is an intermediate representation used for batching multiple trees
    into tensor form.

    Attributes:
        component_id: Index into component embedding (0 for internal nodes)
        operator_id: Index into operator embedding (0 for leaves)
        children: Tuple of child IndexedTrees
        depth: Depth of this subtree
    """
    component_id: int
    operator_id: int
    children: Tuple["IndexedTree", ...] = ()
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    @property
    def arity(self) -> int:
        """Number of children."""
        return len(self.children)


@dataclass
class BatchedTrees:
    """
    Batched representation of multiple trees for efficient GPU processing.

    Trees are flattened into level-ordered arrays, allowing vectorized
    bottom-up computation where all nodes at the same depth level are
    processed simultaneously.

    Attributes:
        component_ids: [num_nodes] Component indices (0 for internal nodes)
        operator_ids: [num_nodes] Operator indices (0 for leaves)
        left_child: [num_nodes] Index of left child (-1 for leaves)
        right_child: [num_nodes] Index of right child (-1 for leaves)
        third_child: [num_nodes] Index of third child for ternary ops (-1 otherwise)
        parent: [num_nodes] Index of parent node (-1 for roots)
        depth: [num_nodes] Depth of each node
        is_leaf: [num_nodes] Boolean mask for leaf nodes
        root_indices: [batch_size] Indices of root nodes
        batch_size: Number of trees in batch
        max_depth: Maximum depth across all trees
        num_nodes: Total number of nodes
    """
    component_ids: torch.LongTensor
    operator_ids: torch.LongTensor
    left_child: torch.LongTensor
    right_child: torch.LongTensor
    third_child: torch.LongTensor
    parent: torch.LongTensor
    depth: torch.LongTensor
    is_leaf: torch.BoolTensor
    root_indices: torch.LongTensor
    batch_size: int
    max_depth: int
    num_nodes: int

    def to(self, device: torch.device) -> "BatchedTrees":
        """Move all tensors to specified device."""
        return BatchedTrees(
            component_ids=self.component_ids.to(device),
            operator_ids=self.operator_ids.to(device),
            left_child=self.left_child.to(device),
            right_child=self.right_child.to(device),
            third_child=self.third_child.to(device),
            parent=self.parent.to(device),
            depth=self.depth.to(device),
            is_leaf=self.is_leaf.to(device),
            root_indices=self.root_indices.to(device),
            batch_size=self.batch_size,
            max_depth=self.max_depth,
            num_nodes=self.num_nodes,
        )

    def get_level_mask(self, level: int) -> torch.BoolTensor:
        """Get mask for nodes at a specific depth level."""
        return self.depth == level

    @classmethod
    def from_trees(
        cls,
        trees: List[IndexedTree],
        device: Optional[torch.device] = None,
    ) -> "BatchedTrees":
        """
        Batch multiple IndexedTrees into a single BatchedTrees structure.

        The trees are flattened in level order (breadth-first), with nodes
        at the same depth level grouped together for efficient processing.

        Args:
            trees: List of IndexedTree objects to batch
            device: Optional device to place tensors on

        Returns:
            BatchedTrees containing all trees
        """
        if not trees:
            raise ValueError("Cannot batch empty list of trees")

        # First pass: count nodes and find max depth
        total_nodes = 0
        max_depth = 0
        for tree in trees:
            total_nodes += _count_nodes(tree)
            max_depth = max(max_depth, tree.depth)

        # Allocate arrays
        component_ids = torch.zeros(total_nodes, dtype=torch.long)
        operator_ids = torch.zeros(total_nodes, dtype=torch.long)
        left_child = torch.full((total_nodes,), -1, dtype=torch.long)
        right_child = torch.full((total_nodes,), -1, dtype=torch.long)
        third_child = torch.full((total_nodes,), -1, dtype=torch.long)
        parent = torch.full((total_nodes,), -1, dtype=torch.long)
        depth = torch.zeros(total_nodes, dtype=torch.long)
        is_leaf = torch.zeros(total_nodes, dtype=torch.bool)
        root_indices = torch.zeros(len(trees), dtype=torch.long)

        # Second pass: flatten trees
        node_idx = 0
        for tree_idx, tree in enumerate(trees):
            root_idx = node_idx
            root_indices[tree_idx] = root_idx
            node_idx = _flatten_tree(
                tree,
                node_idx,
                -1,  # No parent for root
                component_ids,
                operator_ids,
                left_child,
                right_child,
                third_child,
                parent,
                depth,
                is_leaf,
            )

        if device is not None:
            component_ids = component_ids.to(device)
            operator_ids = operator_ids.to(device)
            left_child = left_child.to(device)
            right_child = right_child.to(device)
            third_child = third_child.to(device)
            parent = parent.to(device)
            depth = depth.to(device)
            is_leaf = is_leaf.to(device)
            root_indices = root_indices.to(device)

        return cls(
            component_ids=component_ids,
            operator_ids=operator_ids,
            left_child=left_child,
            right_child=right_child,
            third_child=third_child,
            parent=parent,
            depth=depth,
            is_leaf=is_leaf,
            root_indices=root_indices,
            batch_size=len(trees),
            max_depth=max_depth,
            num_nodes=total_nodes,
        )


def _count_nodes(tree: IndexedTree) -> int:
    """Count total nodes in an IndexedTree."""
    if tree.is_leaf:
        return 1
    return 1 + sum(_count_nodes(child) for child in tree.children)


def _flatten_tree(
    tree: IndexedTree,
    node_idx: int,
    parent_idx: int,
    component_ids: torch.Tensor,
    operator_ids: torch.Tensor,
    left_child: torch.Tensor,
    right_child: torch.Tensor,
    third_child: torch.Tensor,
    parent: torch.Tensor,
    depth: torch.Tensor,
    is_leaf: torch.Tensor,
) -> int:
    """
    Flatten a tree into the pre-allocated arrays.

    Returns the next available node index.
    """
    current_idx = node_idx
    node_idx += 1

    # Set node properties
    component_ids[current_idx] = tree.component_id
    operator_ids[current_idx] = tree.operator_id
    parent[current_idx] = parent_idx
    depth[current_idx] = tree.depth
    is_leaf[current_idx] = tree.is_leaf

    if not tree.is_leaf:
        # Process children and record their indices
        child_indices = []
        for child in tree.children:
            child_idx = node_idx
            child_indices.append(child_idx)
            node_idx = _flatten_tree(
                child,
                node_idx,
                current_idx,
                component_ids,
                operator_ids,
                left_child,
                right_child,
                third_child,
                parent,
                depth,
                is_leaf,
            )

        # Record child indices
        if len(child_indices) >= 1:
            left_child[current_idx] = child_indices[0]
        if len(child_indices) >= 2:
            right_child[current_idx] = child_indices[1]
        if len(child_indices) >= 3:
            third_child[current_idx] = child_indices[2]

    return node_idx


def create_empty_batched_trees(device: Optional[torch.device] = None) -> BatchedTrees:
    """Create an empty BatchedTrees for edge cases."""
    return BatchedTrees(
        component_ids=torch.tensor([], dtype=torch.long, device=device),
        operator_ids=torch.tensor([], dtype=torch.long, device=device),
        left_child=torch.tensor([], dtype=torch.long, device=device),
        right_child=torch.tensor([], dtype=torch.long, device=device),
        third_child=torch.tensor([], dtype=torch.long, device=device),
        parent=torch.tensor([], dtype=torch.long, device=device),
        depth=torch.tensor([], dtype=torch.long, device=device),
        is_leaf=torch.tensor([], dtype=torch.bool, device=device),
        root_indices=torch.tensor([], dtype=torch.long, device=device),
        batch_size=0,
        max_depth=0,
        num_nodes=0,
    )
