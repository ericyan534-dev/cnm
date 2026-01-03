"""
Whole Word Masking (WWM) utilities for Chinese text.

Chinese text doesn't have explicit word boundaries, so we use
word segmentation (jieba) to determine masking units.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional, Set

import torch


def get_chinese_word_boundaries(
    text: str,
    segmenter=None,
) -> List[Tuple[int, int]]:
    """
    Get word boundaries in Chinese text.

    Args:
        text: Chinese text string
        segmenter: Word segmenter (jieba-like), uses jieba if None

    Returns:
        List of (start, end) character positions for each word
    """
    if segmenter is None:
        import jieba
        segmenter = jieba

    boundaries = []
    current_pos = 0

    for word in segmenter.cut(text):
        start = current_pos
        end = current_pos + len(word)
        boundaries.append((start, end))
        current_pos = end

    return boundaries


def create_wwm_mask(
    input_ids: torch.LongTensor,
    tokenizer,
    mlm_probability: float = 0.15,
    special_token_ids: Optional[Set[int]] = None,
) -> torch.BoolTensor:
    """
    Create Whole Word Masking mask for a batch.

    Args:
        input_ids: [batch, seq] Token IDs
        tokenizer: Tokenizer for converting IDs to tokens
        mlm_probability: Probability of masking a word
        special_token_ids: Set of special token IDs to never mask

    Returns:
        [batch, seq] Boolean mask where True = mask this position
    """
    import jieba

    if special_token_ids is None:
        special_token_ids = {
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        }
        special_token_ids.discard(None)

    batch_size, seq_length = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(batch_size):
        # Convert to tokens
        ids = input_ids[b].tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # Build text and track token-to-char mapping
        text_chars = []
        token_to_char_start = []
        token_to_char_end = []

        for i, token in enumerate(tokens):
            if ids[i] in special_token_ids:
                token_to_char_start.append(-1)
                token_to_char_end.append(-1)
                continue

            # Handle wordpiece tokens
            if token.startswith('##'):
                token = token[2:]

            start = len(text_chars)
            text_chars.extend(token)
            end = len(text_chars)

            token_to_char_start.append(start)
            token_to_char_end.append(end)

        if not text_chars:
            continue

        text = ''.join(text_chars)

        # Get word boundaries
        word_boundaries = get_chinese_word_boundaries(text)

        # Map char boundaries to token boundaries
        token_boundaries = []
        for word_start, word_end in word_boundaries:
            # Find tokens that overlap with this word
            tok_start = None
            tok_end = None

            for i in range(seq_length):
                t_start = token_to_char_start[i]
                t_end = token_to_char_end[i]

                if t_start == -1:  # Special token
                    continue

                # Check overlap
                if t_start < word_end and t_end > word_start:
                    if tok_start is None:
                        tok_start = i
                    tok_end = i + 1

            if tok_start is not None:
                token_boundaries.append((tok_start, tok_end))

        # Randomly select words to mask
        num_words = len(token_boundaries)
        num_to_mask = max(1, int(num_words * mlm_probability))

        indices = list(range(num_words))
        random.shuffle(indices)

        for idx in indices[:num_to_mask]:
            start, end = token_boundaries[idx]
            mask[b, start:end] = True

    return mask


def apply_wwm_masking(
    input_ids: torch.LongTensor,
    mask: torch.BoolTensor,
    tokenizer,
    vocab_size: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Apply masking to input_ids based on WWM mask.

    80% -> [MASK]
    10% -> random token
    10% -> original

    Args:
        input_ids: [batch, seq] Original token IDs
        mask: [batch, seq] Boolean mask
        tokenizer: Tokenizer for special tokens
        vocab_size: Vocabulary size for random replacement

    Returns:
        Tuple of (masked_input_ids, labels)
    """
    labels = input_ids.clone()
    masked_input_ids = input_ids.clone()

    # Labels: -100 for non-masked positions
    labels[~mask] = -100

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(torch.full_like(mask, 0.8, dtype=torch.float)).bool()
    replace_mask = replace_mask & mask
    masked_input_ids[replace_mask] = tokenizer.mask_token_id

    # 10% -> random (of remaining 20%, so 50% of that)
    random_mask = torch.bernoulli(torch.full_like(mask, 0.5, dtype=torch.float)).bool()
    random_mask = random_mask & mask & ~replace_mask
    random_tokens = torch.randint(0, vocab_size, input_ids.shape, dtype=torch.long)
    masked_input_ids[random_mask] = random_tokens[random_mask]

    # 10% -> keep original (implicit)

    return masked_input_ids, labels


class WholeWordMasker:
    """
    Class for applying Whole Word Masking to Chinese text.

    Usage:
        masker = WholeWordMasker(tokenizer, mlm_probability=0.15)
        masked_ids, labels = masker(input_ids)
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        special_token_ids: Optional[Set[int]] = None,
    ):
        """
        Initialize WholeWordMasker.

        Args:
            tokenizer: Tokenizer instance
            mlm_probability: Probability of masking each word
            special_token_ids: IDs to never mask (auto-detected if None)
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.vocab_size = len(tokenizer)

        if special_token_ids is None:
            self.special_token_ids = {
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id,
            }
            self.special_token_ids.discard(None)
        else:
            self.special_token_ids = special_token_ids

        # Initialize jieba
        import jieba
        jieba.setLogLevel(20)  # Suppress logs
        self.segmenter = jieba

    def __call__(
        self,
        input_ids: torch.LongTensor,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Apply WWM to input_ids.

        Args:
            input_ids: [batch, seq] Token IDs

        Returns:
            Tuple of (masked_input_ids, labels)
        """
        mask = create_wwm_mask(
            input_ids,
            self.tokenizer,
            self.mlm_probability,
            self.special_token_ids,
        )
        return apply_wwm_masking(
            input_ids, mask, self.tokenizer, self.vocab_size
        )
