"""
Data collators for CNM-BERT training.

Provides collators that handle:
- Standard MLM masking
- Whole Word Masking (WWM) for Chinese
- Structural input preparation
- Component label generation for auxiliary loss
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import DataCollatorForLanguageModeling

from cnm.data.tokenizer import CNMTokenizer
from cnm.data.vocab import CNMVocab


@dataclass
class CNMDataCollatorForPreTraining:
    """
    Data collator for CNM-BERT pretraining with WWM and component labels.

    Handles:
    1. Whole Word Masking (WWM) for Chinese text
    2. Standard MLM objective
    3. Structural input preparation (unique_trees, unique_to_position)
    4. Component label generation for auxiliary loss
    """

    tokenizer: CNMTokenizer
    cnm_vocab: Optional[CNMVocab] = None
    mlm_probability: float = 0.15
    wwm: bool = True
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        """Initialize jieba for word segmentation if WWM is enabled."""
        if self.wwm:
            try:
                import jieba
                self.segmenter = jieba
                # Load jieba dictionary silently
                jieba.setLogLevel(20)  # WARNING level
            except ImportError:
                print("Warning: jieba not installed. Falling back to character-level masking.")
                self.wwm = False
                self.segmenter = None

        if self.cnm_vocab is None and hasattr(self.tokenizer, 'cnm_vocab'):
            self.cnm_vocab = self.tokenizer.cnm_vocab

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.

        Args:
            examples: List of dicts with 'input_ids', 'attention_mask', etc.

        Returns:
            Batch dict with all necessary tensors
        """
        # Extract input_ids and attention_mask
        if isinstance(examples[0], dict):
            input_ids = [e['input_ids'] for e in examples]
            attention_mask = [e.get('attention_mask') for e in examples]
        else:
            input_ids = examples
            attention_mask = None

        # Pad sequences
        batch = self._pad_sequences(input_ids, attention_mask)

        # Apply masking
        batch['input_ids'], batch['labels'] = self._mask_tokens(
            batch['input_ids'].clone()
        )

        # Prepare structural inputs
        if self.cnm_vocab is not None:
            unique_trees, unique_to_position = self.tokenizer.prepare_structural_inputs(
                batch['input_ids']
            )
            batch['unique_trees'] = unique_trees
            batch['unique_to_position'] = unique_to_position

            # Generate component labels for auxiliary loss
            batch['component_labels'] = self._generate_component_labels(
                batch['input_ids'], batch['labels']
            )

        return batch

    def _pad_sequences(
        self,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]],
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to same length."""
        # Find max length
        max_len = max(len(ids) for ids in input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        max_len = min(max_len, self.max_length)

        # Pad
        padded_input_ids = []
        padded_attention_mask = []

        pad_token_id = self.tokenizer.pad_token_id or 0

        for i, ids in enumerate(input_ids):
            # Truncate if necessary
            ids = ids[:max_len]
            padding_length = max_len - len(ids)

            padded_input_ids.append(ids + [pad_token_id] * padding_length)

            if attention_mask is not None and attention_mask[i] is not None:
                mask = attention_mask[i][:max_len]
                padded_attention_mask.append(mask + [0] * padding_length)
            else:
                padded_attention_mask.append([1] * len(ids) + [0] * padding_length)

        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
        }

    def _mask_tokens(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking with optional WWM.

        Args:
            input_ids: [batch, seq] Token IDs

        Returns:
            Tuple of (masked_input_ids, labels)
        """
        labels = input_ids.clone()
        batch_size, seq_length = input_ids.shape

        # Special tokens mask
        special_tokens_mask = self._get_special_tokens_mask(input_ids)

        if self.wwm:
            # Whole Word Masking
            masked_indices = self._get_wwm_mask(input_ids, special_tokens_mask)
        else:
            # Standard random masking
            probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels: -100 for non-masked tokens
        labels[~masked_indices] = -100

        # 80% of time: replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, 0.8)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time: replace with random token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% of time: keep original (already done)

        return input_ids, labels

    def _get_special_tokens_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for special tokens."""
        special_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.mask_token_id,
        }
        special_ids.discard(None)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in special_ids:
            mask |= (input_ids == special_id)

        return mask

    def _get_wwm_mask(
        self,
        input_ids: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get Whole Word Masking mask.

        Groups characters into words using jieba, then masks entire words.
        """
        batch_size, seq_length = input_ids.shape
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)

        for b in range(batch_size):
            # Convert tokens back to text for segmentation
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].tolist())

            # Find word boundaries
            word_boundaries = self._get_word_boundaries(tokens)

            # Decide which words to mask
            num_words = len(word_boundaries)
            num_to_mask = max(1, int(num_words * self.mlm_probability))

            # Randomly select words to mask
            word_indices = list(range(num_words))
            random.shuffle(word_indices)

            for word_idx in word_indices[:num_to_mask]:
                start, end = word_boundaries[word_idx]
                # Check if any token in word is special
                if not special_tokens_mask[b, start:end].any():
                    masked_indices[b, start:end] = True

        return masked_indices

    def _get_word_boundaries(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Get word boundaries from tokens using jieba segmentation.

        Returns list of (start, end) tuples for each word.
        """
        # Reconstruct text (handle ## wordpieces)
        text_parts = []
        token_starts = []
        current_pos = 0

        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']:
                token_starts.append((current_pos, current_pos))
                continue

            if token.startswith('##'):
                token = token[2:]

            text_parts.append(token)
            token_starts.append((current_pos, current_pos + len(token)))
            current_pos += len(token)

        text = ''.join(text_parts)

        if not text:
            return [(i, i + 1) for i in range(len(tokens))]

        # Segment with jieba
        words = list(self.segmenter.cut(text))

        # Map words back to token indices
        boundaries = []
        char_pos = 0
        token_idx = 0

        for word in words:
            word_start = char_pos
            word_end = char_pos + len(word)

            # Find token indices that overlap with this word
            start_token = None
            end_token = None

            for i, (t_start, t_end) in enumerate(token_starts):
                if t_start < word_end and t_end > word_start:
                    if start_token is None:
                        start_token = i
                    end_token = i + 1

            if start_token is not None:
                boundaries.append((start_token, end_token))

            char_pos = word_end

        # Handle any remaining tokens
        if not boundaries:
            boundaries = [(i, i + 1) for i in range(len(tokens))]

        return boundaries

    def _generate_component_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate component labels for auxiliary loss.

        Args:
            input_ids: [batch, seq] Token IDs
            labels: [batch, seq] MLM labels (-100 for non-masked)

        Returns:
            [batch, seq, component_vocab_size] Multi-hot labels
        """
        batch_size, seq_length = input_ids.shape
        component_vocab_size = self.cnm_vocab.component_vocab_size

        component_labels = torch.zeros(
            batch_size, seq_length, component_vocab_size,
            dtype=torch.float
        )

        for b in range(batch_size):
            for s in range(seq_length):
                if labels[b, s] != -100:  # Masked position
                    original_token_id = labels[b, s].item()
                    if original_token_id in self.tokenizer.token_to_char:
                        char = self.tokenizer.token_to_char[original_token_id]
                        char_labels = self.cnm_vocab.get_component_labels(char)
                        component_labels[b, s] = char_labels

        return component_labels


@dataclass
class CNMDataCollatorForFineTuning:
    """
    Data collator for CNM-BERT fine-tuning (no masking).

    Only prepares structural inputs without applying MLM.
    """

    tokenizer: CNMTokenizer
    cnm_vocab: Optional[CNMVocab] = None
    max_length: int = 512
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.cnm_vocab is None and hasattr(self.tokenizer, 'cnm_vocab'):
            self.cnm_vocab = self.tokenizer.cnm_vocab

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples for fine-tuning."""
        # Pad sequences
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Prepare structural inputs
        if self.cnm_vocab is not None:
            unique_trees, unique_to_position = self.tokenizer.prepare_structural_inputs(
                batch['input_ids']
            )
            batch['unique_trees'] = unique_trees
            batch['unique_to_position'] = unique_to_position

        # Handle labels if present
        if 'label' in examples[0]:
            batch['labels'] = torch.tensor([e['label'] for e in examples])
        elif 'labels' in examples[0]:
            batch['labels'] = torch.tensor([e['labels'] for e in examples])

        return batch
