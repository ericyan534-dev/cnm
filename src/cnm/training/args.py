"""
Extended training arguments for CNM-BERT.

Adds CNM-specific parameters to HuggingFace TrainingArguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class CNMTrainingArguments(TrainingArguments):
    """
    Training arguments for CNM-BERT.

    Extends TrainingArguments with CNM-specific parameters.
    """

    # CNM-specific parameters
    struct_dim: int = field(
        default=256,
        metadata={"help": "Dimension of structural embeddings"},
    )

    tree_hidden_dim: int = field(
        default=512,
        metadata={"help": "Hidden dimension in Tree-MLP"},
    )

    max_tree_depth: int = field(
        default=6,
        metadata={"help": "Maximum depth for IDS tree parsing"},
    )

    aux_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for auxiliary component prediction loss"},
    )

    # WWM parameters
    wwm: bool = field(
        default=True,
        metadata={"help": "Use Whole Word Masking for Chinese"},
    )

    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Probability of masking tokens in MLM"},
    )

    # Data parameters
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"},
    )

    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data preprocessing"},
    )

    # Model initialization
    pretrained_bert: Optional[str] = field(
        default="bert-base-chinese",
        metadata={"help": "Pretrained BERT model to initialize from"},
    )

    cnm_vocab_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CNM vocabulary file"},
    )

    ids_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to IDS parse cache"},
    )

    # Logging
    log_mlm_loss: bool = field(
        default=True,
        metadata={"help": "Log MLM loss separately"},
    )

    log_aux_loss: bool = field(
        default=True,
        metadata={"help": "Log auxiliary loss separately"},
    )


@dataclass
class CNMPretrainingArguments(CNMTrainingArguments):
    """
    Arguments specifically for CNM-BERT pretraining.
    """

    # Pretraining data
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Training data file or directory"},
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Validation data file"},
    )

    validation_split_percentage: int = field(
        default=5,
        metadata={"help": "Percentage of training data to use for validation"},
    )

    # Training schedule
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples"},
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples"},
    )

    # Optimization
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for pretraining"},
    )

    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"},
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"},
    )

    # Default W&B logging
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to wandb by default"},
    )


@dataclass
class CNMFinetuningArguments(CNMTrainingArguments):
    """
    Arguments specifically for CNM-BERT fine-tuning.
    """

    # Task
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the CLUE task to fine-tune on"},
    )

    # Model
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained CNM-BERT model"},
    )

    # Fine-tuning parameters
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for fine-tuning"},
    )

    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Number of training epochs"},
    )

    per_device_train_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per device for training"},
    )

    per_device_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per device for evaluation"},
    )

    # Evaluation
    evaluation_strategy: str = field(
        default="epoch",
        metadata={"help": "Evaluation strategy"},
    )

    save_strategy: str = field(
        default="epoch",
        metadata={"help": "Save strategy"},
    )

    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Load best model at end of training"},
    )

    metric_for_best_model: str = field(
        default="accuracy",
        metadata={"help": "Metric for selecting best model"},
    )
