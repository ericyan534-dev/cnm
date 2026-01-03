"""Training module for CNM-BERT."""

from cnm.training.wwm import WholeWordMasker, create_wwm_mask
from cnm.training.losses import MLMLoss, ComponentPredictionLoss, CNMPretrainingLoss
from cnm.training.trainer import CNMTrainer, CNMTrainerForSequenceClassification
from cnm.training.args import CNMTrainingArguments, CNMPretrainingArguments, CNMFinetuningArguments

__all__ = [
    "WholeWordMasker",
    "create_wwm_mask",
    "MLMLoss",
    "ComponentPredictionLoss",
    "CNMPretrainingLoss",
    "CNMTrainer",
    "CNMTrainerForSequenceClassification",
    "CNMTrainingArguments",
    "CNMPretrainingArguments",
    "CNMFinetuningArguments",
]
