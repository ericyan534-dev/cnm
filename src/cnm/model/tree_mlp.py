"""
Tree-MLP Encoder for Chinese character structural embeddings.

This module implements a recursive neural network that encodes Chinese characters
based on their compositional structure (IDS decomposition). The key innovation
is vectorized bottom-up computation that processes all nodes at the same depth
level simultaneously, avoiding slow recursive function calls.

Architecture:
    h = LayerNorm(MLP([e_op; h_left; h_right]) + h_left + h_right)

where:
    - e_op: embedding of the IDC operator
    - h_left, h_right: embeddings of child nodes
    - MLP: two-layer feed-forward network
    - Residual connection adds child embeddings for gradient flow
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnm.data.tree import BatchedTrees


class TreeMLPEncoder(nn.Module):
    """
    Vectorized Tree-MLP encoder for Chinese character structure.

    This encoder computes embeddings for Chinese characters based on their
    IDS decomposition trees. It uses a bottom-up computation strategy:
    1. Initialize leaf embeddings from component vocabulary
    2. For each depth level (from leaves to root):
       - Gather child embeddings
       - Compute parent embedding: h = LN(MLP([op; left; right]) + left + right)
    3. Return root embeddings

    Attributes:
        component_embed: Embedding layer for atomic components
        operator_embed: Embedding layer for IDC operators
        combine_mlp: MLP for combining children with operator
        layer_norm: Layer normalization for residual connection
    """

    def __init__(
        self,
        component_vocab_size: int,
        operator_vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        max_depth: int = 6,
        dropout: float = 0.1,
    ):
        """
        Initialize the Tree-MLP encoder.

        Args:
            component_vocab_size: Size of component vocabulary
            operator_vocab_size: Size of operator vocabulary (typically 14-16)
            embed_dim: Dimension of embeddings (default: 256)
            hidden_dim: Hidden dimension of MLP (default: 512)
            max_depth: Maximum tree depth (default: 6)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth

        # Embedding layers
        self.component_embed = nn.Embedding(
            component_vocab_size, embed_dim, padding_idx=0
        )
        self.operator_embed = nn.Embedding(
            operator_vocab_size, embed_dim, padding_idx=0
        )

        # MLP for binary operators: [op; left; right] -> embed_dim
        # Input: 3 * embed_dim (operator + 2 children)
        self.binary_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # MLP for ternary operators: [op; left; mid; right] -> embed_dim
        # Input: 4 * embed_dim (operator + 3 children)
        self.ternary_mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Layer normalization for residual
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(self, batched_trees: BatchedTrees) -> torch.Tensor:
        """
        Compute embeddings for a batch of character trees.

        This method performs vectorized bottom-up computation:
        1. Initialize all node embeddings (leaves from component_embed)
        2. Process depth levels from max_depth-1 down to 0
        3. At each level, compute parent embeddings from children

        Args:
            batched_trees: BatchedTrees structure containing flattened trees

        Returns:
            [batch_size, embed_dim] tensor of root embeddings
        """
        # Get the device of model weights (canonical device for this replica)
        model_device = self.component_embed.weight.device

        if batched_trees.num_nodes == 0:
            return torch.zeros(
                batched_trees.batch_size,
                self.embed_dim,
                device=model_device,
            )

        # CRITICAL: Move batched_trees tensors to model device to avoid device mismatch
        # This is necessary for DataParallel where replicas run on different GPUs
        batched_trees = batched_trees.to(model_device)

        device = model_device
        num_nodes = batched_trees.num_nodes

        # Initialize node embeddings
        # Leaves get component embeddings, internal nodes will be computed
        node_embeds = self.component_embed(batched_trees.component_ids)

        # Process depth levels bottom-up
        for level in range(batched_trees.max_depth - 1, -1, -1):
            level_mask = batched_trees.get_level_mask(level)
            if not level_mask.any():
                continue

            # Get indices of nodes at this level that are internal (have children)
            internal_mask = level_mask & ~batched_trees.is_leaf

            if not internal_mask.any():
                continue

            # Get operator embeddings for internal nodes
            op_ids = batched_trees.operator_ids[internal_mask]
            op_embeds = self.operator_embed(op_ids)

            # Get child indices
            left_idx = batched_trees.left_child[internal_mask]
            right_idx = batched_trees.right_child[internal_mask]
            third_idx = batched_trees.third_child[internal_mask]

            # Get child embeddings
            left_embeds = node_embeds[left_idx]
            right_embeds = node_embeds[right_idx]

            # Determine which nodes have ternary operators
            has_third = third_idx >= 0

            if has_third.any():
                # Process ternary nodes
                ternary_mask = has_third
                binary_mask = ~has_third

                if binary_mask.any():
                    # Binary nodes
                    binary_combined = torch.cat([
                        op_embeds[binary_mask],
                        left_embeds[binary_mask],
                        right_embeds[binary_mask],
                    ], dim=-1)
                    binary_output = self.binary_mlp(binary_combined)
                    binary_residual = left_embeds[binary_mask] + right_embeds[binary_mask]
                    binary_embeds = self.layer_norm(binary_output + binary_residual)

                if ternary_mask.any():
                    # Ternary nodes
                    third_embeds = node_embeds[third_idx[ternary_mask]]
                    ternary_combined = torch.cat([
                        op_embeds[ternary_mask],
                        left_embeds[ternary_mask],
                        right_embeds[ternary_mask],
                        third_embeds,
                    ], dim=-1)
                    ternary_output = self.ternary_mlp(ternary_combined)
                    ternary_residual = (
                        left_embeds[ternary_mask] +
                        right_embeds[ternary_mask] +
                        third_embeds
                    ) / 3.0  # Average for 3 children
                    ternary_embeds = self.layer_norm(ternary_output + ternary_residual)

                # Combine results
                new_embeds = torch.zeros(
                    internal_mask.sum(), self.embed_dim, device=device
                )
                if binary_mask.any():
                    new_embeds[binary_mask] = binary_embeds
                if ternary_mask.any():
                    new_embeds[ternary_mask] = ternary_embeds

                node_embeds[internal_mask] = new_embeds
            else:
                # All binary - simpler path
                combined = torch.cat([op_embeds, left_embeds, right_embeds], dim=-1)
                output = self.binary_mlp(combined)
                residual = left_embeds + right_embeds
                new_embeds = self.layer_norm(output + residual)
                node_embeds[internal_mask] = new_embeds

        # Extract root embeddings
        root_embeds = node_embeds[batched_trees.root_indices]
        return self.dropout(root_embeds)

    def forward_single(self, batched_trees: BatchedTrees, index: int) -> torch.Tensor:
        """
        Compute embedding for a single tree in the batch.

        Useful for debugging and analysis.

        Args:
            batched_trees: BatchedTrees structure
            index: Index of the tree to compute

        Returns:
            [embed_dim] tensor of the root embedding
        """
        all_embeds = self.forward(batched_trees)
        return all_embeds[index]

    def get_component_embedding(self, component_id: int) -> torch.Tensor:
        """Get the embedding for a component by ID."""
        ids = torch.tensor([component_id], device=self.component_embed.weight.device)
        return self.component_embed(ids).squeeze(0)

    def get_operator_embedding(self, operator_id: int) -> torch.Tensor:
        """Get the embedding for an operator by ID."""
        ids = torch.tensor([operator_id], device=self.operator_embed.weight.device)
        return self.operator_embed(ids).squeeze(0)


class TreeMLPEncoderWithCache(TreeMLPEncoder):
    """
    Tree-MLP encoder with embedding cache for efficiency.

    For inference and during training with repeated characters, this version
    caches computed embeddings to avoid redundant computation.
    """

    def __init__(self, *args, cache_size: int = 10000, **kwargs):
        """
        Initialize with cache.

        Args:
            cache_size: Maximum number of embeddings to cache
            *args, **kwargs: Passed to TreeMLPEncoder
        """
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self._cache: dict = {}
        self._cache_order: list = []

    def forward_with_cache(
        self,
        batched_trees: BatchedTrees,
        char_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Forward pass with caching.

        Args:
            batched_trees: BatchedTrees structure
            char_ids: Optional list of character identifiers for caching

        Returns:
            [batch_size, embed_dim] tensor of embeddings
        """
        if char_ids is None or self.training:
            # No caching during training or if no IDs provided
            return self.forward(batched_trees)

        device = batched_trees.component_ids.device
        batch_size = batched_trees.batch_size
        result = torch.zeros(batch_size, self.embed_dim, device=device)

        # Find which are cached vs need computation
        need_compute = []
        need_compute_indices = []

        for i, char_id in enumerate(char_ids):
            if char_id in self._cache:
                result[i] = self._cache[char_id].to(device)
            else:
                need_compute.append(i)
                need_compute_indices.append(i)

        if need_compute:
            # Compute embeddings for uncached characters
            # This would require subsetting the BatchedTrees, which is complex
            # For now, just compute all and cache the results
            computed = self.forward(batched_trees)

            for i, char_id in enumerate(char_ids):
                if char_id not in self._cache:
                    self._add_to_cache(char_id, computed[i].detach().cpu())
                    result[i] = computed[i]

        return result

    def _add_to_cache(self, char_id: str, embedding: torch.Tensor):
        """Add an embedding to the cache."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[char_id] = embedding
        self._cache_order.append(char_id)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_order.clear()
