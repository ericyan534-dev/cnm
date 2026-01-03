"""Evaluation module for CNM-BERT."""

from cnm.evaluation.clue import (
    CLUEEvaluator,
    CLUE_TASKS,
    get_task_config,
    load_clue_dataset,
    compute_clue_metrics,
)

__all__ = [
    "CLUEEvaluator",
    "CLUE_TASKS",
    "get_task_config",
    "load_clue_dataset",
    "compute_clue_metrics",
]
