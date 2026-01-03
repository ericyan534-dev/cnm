"""
Fused embeddings for CNM-BERT.

This module implements CNMEmbeddings, which combines standard BERT embeddings
with structural embeddings from the Tree-MLP encoder. The fusion uses an
Identity+Zero initialization strategy to preserve pretrained BERT weights.

Fusion equation:
    E_final = Project(Concat(E_bert, E_struct))

where Project is initialized to preserve E_bert initially.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cnm.data.tree import BatchedTrees
from cnm.model.tree_mlp import TreeMLPEncoder
from cnm.model.configuration_cnm import CNMConfig


class CNMEmbeddings(nn.Module):
    """
    Fused embedding layer combining BERT and structural embeddings.

    This layer:
    1. Computes standard BERT embeddings (word + position + token_type)
    2. Computes structural embeddings via Tree-MLP (on unique characters)
    3. Scatters structural embeddings to match sequence positions
    4. Fuses both streams via a projection layer

    The fusion layer is initialized with Identity+Zero to preserve
    pretrained BERT embeddings at initialization.

    Attributes:
        word_embeddings: Standard BERT word embeddings
        position_embeddings: Position embeddings
        token_type_embeddings: Segment embeddings
        LayerNorm: Layer normalization
        dropout: Dropout layer
        tree_encoder: Tree-MLP for structural embeddings
        fallback_struct: Learnable fallback for non-decomposable tokens
        fusion: Linear layer for fusing BERT + structural embeddings
    """

    def __init__(self, config: CNMConfig, tree_encoder: Optional[TreeMLPEncoder] = None):
        """
        Initialize CNMEmbeddings.

        Args:
            config: CNMConfig with model parameters
            tree_encoder: Pre-built TreeMLPEncoder (or None to create new)
        """
        super().__init__()
        self.config = config

        # Standard BERT embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Structural embedding components
        self.use_struct = config.use_struct_embeddings

        if self.use_struct:
            # Tree-MLP encoder
            if tree_encoder is not None:
                self.tree_encoder = tree_encoder
            else:
                self.tree_encoder = TreeMLPEncoder(
                    component_vocab_size=config.component_vocab_size,
                    operator_vocab_size=config.operator_vocab_size,
                    embed_dim=config.struct_dim,
                    hidden_dim=config.tree_hidden_dim,
                    max_depth=config.max_tree_depth,
                    dropout=config.struct_dropout,
                )

            # Learnable fallback for non-decomposable tokens ([CLS], [SEP], punct, etc.)
            self.fallback_struct = nn.Parameter(torch.zeros(config.struct_dim))

            # Fusion layer: [bert_dim + struct_dim] -> bert_dim
            self.fusion = nn.Linear(
                config.hidden_size + config.struct_dim,
                config.hidden_size,
            )

            # Initialize fusion weights for identity on BERT portion
            self._init_fusion_weights()

        # Register buffer for position ids
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def _init_fusion_weights(self):
        """
        Initialize fusion layer to preserve BERT embeddings.

        Strategy: Identity + Zero
        - Weights for BERT portion: Identity matrix
        - Weights for struct portion: Zero matrix
        - Bias: Zero

        This ensures E_final ≈ E_bert at initialization.
        """
        with torch.no_grad():
            hidden_size = self.config.hidden_size
            struct_dim = self.config.struct_dim

            # Identity for BERT portion
            self.fusion.weight[:, :hidden_size].copy_(
                torch.eye(hidden_size)
            )
            # Zero for structural portion
            self.fusion.weight[:, hidden_size:].zero_()
            # Zero bias
            self.fusion.bias.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor,
        struct_ids: Optional[torch.LongTensor] = None,
        unique_trees: Optional[BatchedTrees] = None,
        unique_to_position: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Compute fused embeddings.

        Args:
            input_ids: [batch, seq] Token IDs
            struct_ids: [batch, seq] Structural token IDs (indices into unique chars)
            unique_trees: BatchedTrees for unique characters in batch
            unique_to_position: [batch, seq] Mapping from positions to unique char indices
            token_type_ids: [batch, seq] Segment IDs
            position_ids: [batch, seq] Position IDs
            inputs_embeds: [batch, seq, hidden] Pre-computed embeddings (optional)

        Returns:
            [batch, seq, hidden_size] Fused embeddings
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        # Position embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        position_embeds = self.position_embeddings(position_ids)

        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = self.token_type_ids[:, :seq_length].expand(batch_size, -1)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine BERT embeddings
        bert_embeds = inputs_embeds + position_embeds + token_type_embeds
        bert_embeds = self.LayerNorm(bert_embeds)

        if not self.use_struct or unique_trees is None:
            # No structural embeddings
            return self.dropout(bert_embeds)

        # Compute structural embeddings
        struct_embeds = self._compute_struct_embeddings(
            unique_trees, unique_to_position, batch_size, seq_length
        )

        # Fuse BERT and structural embeddings
        fused = self.fusion(torch.cat([bert_embeds, struct_embeds], dim=-1))

        return self.dropout(fused)

    def _compute_struct_embeddings(
        self,
        unique_trees: BatchedTrees,
        unique_to_position: Optional[torch.LongTensor],
        batch_size: int,
        seq_length: int,
    ) -> torch.FloatTensor:
        """
        Compute and scatter structural embeddings.

        Args:
            unique_trees: BatchedTrees for unique characters
            unique_to_position: [batch, seq] Mapping to unique char indices
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            [batch, seq, struct_dim] Structural embeddings
        """
        device = self.word_embeddings.weight.device

        # Compute embeddings for unique characters
        if unique_trees.num_nodes > 0:
            unique_embeds = self.tree_encoder(unique_trees)  # [num_unique, struct_dim]
        else:
            unique_embeds = torch.zeros(0, self.config.struct_dim, device=device)

        # Scatter to sequence positions
        if unique_to_position is not None:
            # Create output tensor
            struct_embeds = torch.zeros(
                batch_size, seq_length, self.config.struct_dim, device=device
            )

            # Use scatter to place unique embeddings at correct positions
            # unique_to_position contains -1 for fallback positions
            valid_mask = unique_to_position >= 0

            if valid_mask.any():
                # Flatten for gathering
                flat_indices = unique_to_position[valid_mask]
                gathered = unique_embeds[flat_indices]  # [num_valid, struct_dim]

                # Scatter back
                struct_embeds[valid_mask] = gathered

            # Fill fallback positions
            if (~valid_mask).any():
                struct_embeds[~valid_mask] = self.fallback_struct

        else:
            # No mapping provided - use fallback for all
            struct_embeds = self.fallback_struct.unsqueeze(0).unsqueeze(0).expand(
                batch_size, seq_length, -1
            )

        return struct_embeds

    def load_bert_embeddings(self, bert_embeddings: nn.Module):
        """
        Load pretrained BERT embedding weights.

        Args:
            bert_embeddings: BertEmbeddings module to copy from
        """
        with torch.no_grad():
            self.word_embeddings.weight.copy_(bert_embeddings.word_embeddings.weight)
            self.position_embeddings.weight.copy_(bert_embeddings.position_embeddings.weight)
            self.token_type_embeddings.weight.copy_(bert_embeddings.token_type_embeddings.weight)
            self.LayerNorm.weight.copy_(bert_embeddings.LayerNorm.weight)
            self.LayerNorm.bias.copy_(bert_embeddings.LayerNorm.bias)


class StructEmbeddingProcessor:
    """
    Helper class to prepare structural embeddings for a batch.

    This class handles:
    1. Extracting unique characters from input_ids
    2. Building BatchedTrees for the unique characters
    3. Creating the mapping tensor for scattering

    Usage:
        processor = StructEmbeddingProcessor(tokenizer, vocab)
        unique_trees, mapping = processor.prepare_batch(input_ids)
        embeddings = model.embeddings(input_ids, unique_trees=unique_trees, unique_to_position=mapping)
    """

    def __init__(self, tokenizer, cnm_vocab):
        """
        Initialize processor.

        Args:
            tokenizer: BERT tokenizer for converting IDs to tokens
            cnm_vocab: CNMVocab for building trees
        """
        self.tokenizer = tokenizer
        self.cnm_vocab = cnm_vocab

        # Build mapping from token ID to character (for CJK tokens)
        self._build_token_to_char_map()

    def _build_token_to_char_map(self):
        """Build mapping from token IDs to single characters."""
        self.token_to_char = {}

        for token, idx in self.tokenizer.get_vocab().items():
            # Only single CJK characters
            if len(token) == 1:
                cp = ord(token)
                if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                    self.token_to_char[idx] = token

    def prepare_batch(
        self,
        input_ids: torch.LongTensor,
        device: Optional[torch.device] = None,
    ) -> tuple[BatchedTrees, torch.LongTensor]:
        """
        Prepare structural embeddings for a batch.

        Args:
            input_ids: [batch, seq] Token IDs
            device: Device for output tensors

        Returns:
            Tuple of (BatchedTrees, [batch, seq] mapping tensor)
        """
        batch_size, seq_length = input_ids.shape

        # Find unique CJK characters in batch
        unique_chars = []
        char_to_unique_idx = {}

        # Create mapping tensor (-1 for non-CJK tokens)
        mapping = torch.full(
            (batch_size, seq_length), -1, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            for s in range(seq_length):
                token_id = input_ids[b, s].item()
                if token_id in self.token_to_char:
                    char = self.token_to_char[token_id]

                    if char not in char_to_unique_idx:
                        char_to_unique_idx[char] = len(unique_chars)
                        unique_chars.append(char)

                    mapping[b, s] = char_to_unique_idx[char]

        # Build BatchedTrees for unique characters
        if unique_chars:
            unique_trees = self.cnm_vocab.batch_trees(unique_chars, device=device)
        else:
            from cnm.data.tree import create_empty_batched_trees
            unique_trees = create_empty_batched_trees(device=device)

        return unique_trees, mapping
