"""
Loss functions for CNM-BERT training.

Includes:
- Standard MLM loss
- Component prediction loss (auxiliary)
- Combined CNM pretraining loss
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMLoss(nn.Module):
    """
    Masked Language Modeling loss.

    Standard cross-entropy loss on masked positions only.
    """

    def __init__(self, vocab_size: int, ignore_index: int = -100):
        """
        Initialize MLM loss.

        Args:
            vocab_size: Size of vocabulary
            ignore_index: Label value to ignore (default: -100)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(
        self,
        prediction_scores: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute MLM loss.

        Args:
            prediction_scores: [batch, seq, vocab_size] Logits
            labels: [batch, seq] Target token IDs (-100 for non-masked)

        Returns:
            Scalar loss tensor
        """
        return F.cross_entropy(
            prediction_scores.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )


class ComponentPredictionLoss(nn.Module):
    """
    Auxiliary loss for predicting character components.

    Multi-label binary cross-entropy on masked positions.
    This encourages the model to retain structural information.
    """

    def __init__(
        self,
        component_vocab_size: int,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize component prediction loss.

        Args:
            component_vocab_size: Number of unique components
            pos_weight: Optional positive class weights for imbalanced data
        """
        super().__init__()
        self.component_vocab_size = component_vocab_size
        self.pos_weight = pos_weight

    def forward(
        self,
        component_logits: torch.FloatTensor,
        component_labels: torch.FloatTensor,
        mlm_labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute component prediction loss.

        Only computes loss on masked positions (where mlm_labels != -100).

        Args:
            component_logits: [batch, seq, component_vocab] Logits
            component_labels: [batch, seq, component_vocab] Multi-hot targets
            mlm_labels: [batch, seq] MLM labels to identify masked positions

        Returns:
            Scalar loss tensor
        """
        # Get masked positions
        masked_positions = mlm_labels != -100

        if not masked_positions.any():
            return torch.tensor(0.0, device=component_logits.device)

        # Select masked positions
        masked_logits = component_logits[masked_positions]
        masked_targets = component_labels[masked_positions]

        # Binary cross-entropy with logits
        return F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_targets,
            pos_weight=self.pos_weight,
        )


class CNMPretrainingLoss(nn.Module):
    """
    Combined loss for CNM-BERT pretraining.

    Loss = MLM_loss + aux_weight * Component_loss
    """

    def __init__(
        self,
        vocab_size: int,
        component_vocab_size: int,
        aux_loss_weight: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Initialize combined pretraining loss.

        Args:
            vocab_size: BERT vocabulary size
            component_vocab_size: Number of unique components
            aux_loss_weight: Weight for auxiliary loss
            ignore_index: Label value to ignore
        """
        super().__init__()
        self.mlm_loss = MLMLoss(vocab_size, ignore_index)
        self.component_loss = ComponentPredictionLoss(component_vocab_size)
        self.aux_loss_weight = aux_loss_weight

    def forward(
        self,
        prediction_scores: torch.FloatTensor,
        component_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        component_labels: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Compute combined loss.

        Args:
            prediction_scores: [batch, seq, vocab] MLM logits
            component_logits: [batch, seq, component_vocab] Component logits
            labels: [batch, seq] MLM targets
            component_labels: [batch, seq, component_vocab] Component targets

        Returns:
            Tuple of (total_loss, mlm_loss, aux_loss)
        """
        mlm_loss = self.mlm_loss(prediction_scores, labels)

        if component_labels is not None:
            aux_loss = self.component_loss(component_logits, component_labels, labels)
            total_loss = mlm_loss + self.aux_loss_weight * aux_loss
        else:
            aux_loss = None
            total_loss = mlm_loss

        return total_loss, mlm_loss, aux_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Useful for component prediction where some components are rare.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.FloatTensor,
        targets: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute Focal Loss.

        Args:
            inputs: [*, num_classes] Logits
            targets: [*, num_classes] Binary targets

        Returns:
            Loss tensor
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # p_t = p if y == 1 else 1 - p
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for MLM to improve generalization.
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        """
        Initialize label smoothing loss.

        Args:
            vocab_size: Vocabulary size
            smoothing: Smoothing factor (0 = no smoothing)
            ignore_index: Index to ignore
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        prediction_scores: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute label-smoothed cross entropy.

        Args:
            prediction_scores: [batch, seq, vocab] Logits
            labels: [batch, seq] Targets

        Returns:
            Scalar loss
        """
        # Reshape
        logits = prediction_scores.view(-1, self.vocab_size)
        targets = labels.view(-1)

        # Create smoothed distribution
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (self.vocab_size - 1))
            mask = targets != self.ignore_index
            smooth_labels[mask] = smooth_labels[mask].scatter_(
                1, targets[mask].unsqueeze(1), 1.0 - self.smoothing
            )

        # Compute loss
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        # Mask ignored positions
        loss = loss * mask.float()

        return loss.sum() / mask.sum().clamp(min=1)
