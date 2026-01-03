"""
Dual-stream tokenizer for CNM-BERT.

This tokenizer extends BERT tokenization with structural IDs (struct_ids)
that map each token to its IDS tree representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from cnm.data.vocab import CNMVocab
from cnm.data.tree import BatchedTrees


class CNMTokenizer(BertTokenizerFast):
    """
    Tokenizer for CNM-BERT with dual-stream output.

    Extends BertTokenizerFast to also output struct_ids, which map
    each token position to its structural representation.

    Attributes:
        cnm_vocab: CNMVocab for structural embeddings
        token_to_char: Mapping from token IDs to single CJK characters
        char_to_struct_id: Mapping from characters to structural indices
    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        cnm_vocab: Optional[CNMVocab] = None,
        cnm_vocab_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize CNMTokenizer.

        Args:
            vocab_file: Path to BERT vocabulary file
            cnm_vocab: Pre-built CNMVocab instance
            cnm_vocab_file: Path to saved CNMVocab JSON
            **kwargs: Additional arguments for BertTokenizerFast
        """
        super().__init__(vocab_file=vocab_file, **kwargs)

        # Load CNM vocabulary
        if cnm_vocab is not None:
            self.cnm_vocab = cnm_vocab
        elif cnm_vocab_file is not None:
            self.cnm_vocab = CNMVocab.load(Path(cnm_vocab_file))
        else:
            self.cnm_vocab = None

        # Build token to character mapping
        self._build_token_char_map()

    def _build_token_char_map(self):
        """Build mapping from token IDs to single CJK characters."""
        self.token_to_char: Dict[int, str] = {}
        self.char_to_token_id: Dict[str, int] = {}

        vocab = self.get_vocab()
        for token, idx in vocab.items():
            # Only single CJK characters
            if len(token) == 1:
                cp = ord(token)
                if (0x4E00 <= cp <= 0x9FFF or      # CJK Unified
                    0x3400 <= cp <= 0x4DBF or      # Extension A
                    0x20000 <= cp <= 0x2A6DF or    # Extension B
                    0x2F00 <= cp <= 0x2FDF or      # Kangxi Radicals
                    0x2E80 <= cp <= 0x2EFF):       # Radicals Supplement
                    self.token_to_char[idx] = token
                    self.char_to_token_id[token] = idx

    def __call__(
        self,
        text: Union[str, List[str]],
        return_struct_ids: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize text with optional structural IDs.

        Args:
            text: Input text or list of texts
            return_struct_ids: Whether to include struct_ids
            **kwargs: Additional arguments for tokenization

        Returns:
            BatchEncoding with input_ids, attention_mask, and optionally struct_ids
        """
        # Standard tokenization
        encoding = super().__call__(text, **kwargs)

        # Add structural IDs if requested and CNM vocab is available
        if return_struct_ids and self.cnm_vocab is not None:
            struct_ids = self._compute_struct_ids(encoding['input_ids'])
            encoding['struct_ids'] = struct_ids

        return encoding

    def _compute_struct_ids(
        self,
        input_ids: Union[List[int], List[List[int]], torch.Tensor],
    ) -> Union[List[int], List[List[int]]]:
        """
        Compute structural IDs for input tokens.

        For CJK characters with IDS decomposition, returns the index
        into the unique character list. For other tokens, returns -1.

        Args:
            input_ids: Token IDs from tokenization

        Returns:
            Structural IDs matching input shape
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # Handle batched input
        if isinstance(input_ids[0], list):
            return [self._compute_struct_ids_single(ids) for ids in input_ids]

        return self._compute_struct_ids_single(input_ids)

    def _compute_struct_ids_single(self, input_ids: List[int]) -> List[int]:
        """Compute struct_ids for a single sequence."""
        struct_ids = []

        for token_id in input_ids:
            if token_id in self.token_to_char:
                char = self.token_to_char[token_id]
                # Check if we have a tree for this character
                if char in self.cnm_vocab.char_to_indexed:
                    struct_ids.append(token_id)  # Use token_id as struct_id
                else:
                    struct_ids.append(-1)  # No tree, use fallback
            else:
                struct_ids.append(-1)  # Non-CJK token, use fallback

        return struct_ids

    def prepare_structural_inputs(
        self,
        input_ids: torch.LongTensor,
        device: Optional[torch.device] = None,
    ) -> tuple:
        """
        Prepare structural inputs for model forward pass.

        Args:
            input_ids: [batch, seq] Token IDs
            device: Device for output tensors

        Returns:
            Tuple of (unique_trees: BatchedTrees, unique_to_position: Tensor)
        """
        if self.cnm_vocab is None:
            raise ValueError("CNM vocabulary not loaded")

        batch_size, seq_length = input_ids.shape
        if device is None:
            device = input_ids.device

        # Find unique CJK characters
        unique_chars: List[str] = []
        char_to_unique_idx: Dict[str, int] = {}

        # Create mapping tensor
        unique_to_position = torch.full(
            (batch_size, seq_length), -1, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            for s in range(seq_length):
                token_id = input_ids[b, s].item()
                if token_id in self.token_to_char:
                    char = self.token_to_char[token_id]
                    if char in self.cnm_vocab.char_to_indexed:
                        if char not in char_to_unique_idx:
                            char_to_unique_idx[char] = len(unique_chars)
                            unique_chars.append(char)
                        unique_to_position[b, s] = char_to_unique_idx[char]

        # Build BatchedTrees
        if unique_chars:
            unique_trees = self.cnm_vocab.batch_trees(unique_chars, device=device)
        else:
            from cnm.data.tree import create_empty_batched_trees
            unique_trees = create_empty_batched_trees(device=device)

        return unique_trees, unique_to_position

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save tokenizer and CNM vocabulary."""
        save_directory = Path(save_directory)
        super().save_pretrained(str(save_directory), **kwargs)

        # Save CNM vocabulary
        if self.cnm_vocab is not None:
            cnm_vocab_path = save_directory / 'cnm_vocab.json'
            self.cnm_vocab.save(cnm_vocab_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        cnm_vocab_file: Optional[str] = None,
        **kwargs,
    ) -> "CNMTokenizer":
        """
        Load tokenizer from pretrained.

        Args:
            pretrained_model_name_or_path: Model name or path
            cnm_vocab_file: Path to CNM vocabulary (auto-detected if not provided)
            **kwargs: Additional arguments

        Returns:
            CNMTokenizer instance
        """
        path = Path(pretrained_model_name_or_path)

        # Auto-detect CNM vocab
        if cnm_vocab_file is None and path.is_dir():
            potential_path = path / 'cnm_vocab.json'
            if potential_path.exists():
                cnm_vocab_file = str(potential_path)

        return cls(
            vocab_file=str(path / 'vocab.txt') if path.is_dir() else None,
            cnm_vocab_file=cnm_vocab_file,
            **kwargs,
        )


def create_cnm_tokenizer(
    bert_tokenizer_name: str = 'bert-base-chinese',
    cnm_vocab: Optional[CNMVocab] = None,
) -> CNMTokenizer:
    """
    Create a CNMTokenizer from a pretrained BERT tokenizer.

    Args:
        bert_tokenizer_name: HuggingFace tokenizer name
        cnm_vocab: CNMVocab instance

    Returns:
        CNMTokenizer instance
    """
    # Load base tokenizer
    base_tokenizer = BertTokenizerFast.from_pretrained(bert_tokenizer_name)

    # Create CNM tokenizer with same vocab
    tokenizer = CNMTokenizer(
        vocab_file=None,
        cnm_vocab=cnm_vocab,
    )

    # Copy vocab from base tokenizer
    tokenizer._tokenizer = base_tokenizer._tokenizer

    # Rebuild token-char map
    tokenizer._build_token_char_map()

    return tokenizer
