"""
Custom trainer for CNM-BERT.

Extends HuggingFace Trainer with:
- Structural input handling
- Multi-loss logging (MLM + auxiliary)
- Gradient checkpointing support
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer


class CNMTrainer(Trainer):
    """
    Trainer for CNM-BERT with structural embedding support.

    Handles:
    - Preparing structural inputs (BatchedTrees) during training
    - Logging separate MLM and auxiliary losses
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        cnm_vocab: Optional[CNMVocab] = None,
        tokenizer: Optional[CNMTokenizer] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        **kwargs,
    ):
        """
        Initialize CNMTrainer.

        Args:
            model: CNM-BERT model
            args: Training arguments
            cnm_vocab: CNM vocabulary for structural embeddings
            tokenizer: CNM tokenizer
            data_collator: Data collator
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Metrics computation function
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs,
        )
        self.cnm_vocab = cnm_vocab

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute loss for CNM-BERT.

        Handles structural inputs and multi-loss tracking.
        """
        # Prepare inputs
        labels = inputs.pop("labels", None)
        component_labels = inputs.pop("component_labels", None)

        # Forward pass
        outputs = model(**inputs)

        if labels is not None:
            # Get loss from model output
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Compute loss manually if needed
                from cnm.training.losses import CNMPretrainingLoss
                loss_fn = CNMPretrainingLoss(
                    vocab_size=model.config.vocab_size,
                    component_vocab_size=model.config.component_vocab_size,
                    aux_loss_weight=model.config.aux_loss_weight,
                )
                loss, mlm_loss, aux_loss = loss_fn(
                    outputs.prediction_logits if hasattr(outputs, 'prediction_logits') else outputs.logits,
                    outputs.component_logits if hasattr(outputs, 'component_logits') else None,
                    labels,
                    component_labels,
                )

            # Log individual losses
            if hasattr(outputs, 'mlm_loss') and outputs.mlm_loss is not None:
                self._log_loss('mlm_loss', outputs.mlm_loss)
            if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                self._log_loss('aux_loss', outputs.aux_loss)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _log_loss(self, name: str, value: torch.Tensor):
        """Log a loss value."""
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({name: value.detach().item()})

    def _prepare_inputs(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for the model.

        Moves tensors to device and handles BatchedTrees.
        """
        inputs = super()._prepare_inputs(inputs)

        # Handle BatchedTrees (not a tensor, needs special handling)
        if 'unique_trees' in inputs and inputs['unique_trees'] is not None:
            inputs['unique_trees'] = inputs['unique_trees'].to(self.args.device)

        return inputs

    def create_optimizer(self):
        """
        Create optimizer with separate learning rates for different parameter groups.

        - BERT parameters: base learning rate
        - Tree encoder parameters: potentially higher learning rate
        - Fusion layer: potentially lower learning rate
        """
        if self.optimizer is not None:
            return self.optimizer

        # Get parameter groups
        decay_params = []
        no_decay_params = []
        tree_params = []

        no_decay_names = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'tree_encoder' in name:
                tree_params.append(param)
            elif any(nd in name for nd in no_decay_names):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate,
            },
            {
                'params': tree_params,
                'weight_decay': self.args.weight_decay,
                'lr': self.args.learning_rate * 2,  # Higher LR for tree encoder
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        """
        Save model, tokenizer, and CNM vocabulary.
        """
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # Save CNM vocabulary
        if self.cnm_vocab is not None:
            from pathlib import Path
            vocab_path = Path(output_dir) / 'cnm_vocab.json'
            self.cnm_vocab.save(vocab_path)


class CNMTrainerForSequenceClassification(CNMTrainer):
    """
    Trainer for CNM-BERT sequence classification tasks.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute classification loss.
        """
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)

        if labels is not None:
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                if model.num_labels == 1:
                    loss = nn.functional.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = nn.functional.cross_entropy(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        return (loss, outputs) if return_outputs else loss


def compute_metrics_for_classification(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for classification tasks.

    Args:
        eval_pred: Evaluation predictions

    Returns:
        Dict with accuracy and F1 scores
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = predictions.argmax(axis=-1)

    # Accuracy
    accuracy = (preds == labels).mean()

    # F1 (macro)
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average='macro')

    return {
        'accuracy': accuracy,
        'f1': f1,
    }
