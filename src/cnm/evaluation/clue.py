"""
CLUE benchmark evaluation for CNM-BERT.

Supports the following CLUE tasks:
- AFQMC: Sentence Pair Similarity
- TNEWS: News Classification
- IFLYTEK: App Description Classification
- CMNLI: Natural Language Inference
- CSL: Keyword Recognition
- WSC: Coreference Resolution
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import EvalPrediction

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Task configurations
CLUE_TASKS = {
    'afqmc': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
    },
    'tnews': {
        'num_labels': 15,
        'metric': 'accuracy',
        'text_columns': ['sentence'],
        'label_column': 'label',
    },
    'iflytek': {
        'num_labels': 119,
        'metric': 'accuracy',
        'text_columns': ['sentence'],
        'label_column': 'label',
    },
    'cmnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_columns': ['sentence1', 'sentence2'],
        'label_column': 'label',
    },
    'csl': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['abst', 'keyword'],
        'label_column': 'label',
    },
    'wsc': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_columns': ['text'],
        'label_column': 'label',
    },
}


@dataclass
class CLUETaskConfig:
    """Configuration for a CLUE task."""
    name: str
    num_labels: int
    metric: str
    text_columns: List[str]
    label_column: str


def get_task_config(task_name: str) -> CLUETaskConfig:
    """Get configuration for a CLUE task."""
    if task_name not in CLUE_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(CLUE_TASKS.keys())}")

    config = CLUE_TASKS[task_name]
    return CLUETaskConfig(
        name=task_name,
        num_labels=config['num_labels'],
        metric=config['metric'],
        text_columns=config['text_columns'],
        label_column=config['label_column'],
    )


def load_clue_dataset(
    task_name: str,
    split: str = 'train',
    data_dir: Optional[str] = None,
) -> Dataset:
    """
    Load a CLUE dataset.

    Args:
        task_name: Name of the CLUE task
        split: Dataset split ('train', 'validation', 'test')
        data_dir: Optional local data directory

    Returns:
        HuggingFace Dataset
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")

    # Map split names
    split_map = {
        'train': 'train',
        'validation': 'validation',
        'valid': 'validation',
        'dev': 'validation',
        'test': 'test',
    }
    split = split_map.get(split, split)

    try:
        # Try loading from HuggingFace Hub
        dataset = load_dataset('clue', task_name, split=split)
    except Exception:
        # Try local loading
        if data_dir is not None:
            dataset = load_dataset(
                'json',
                data_files=str(Path(data_dir) / task_name / f'{split}.json'),
                split='train',
            )
        else:
            raise ValueError(f"Could not load CLUE task {task_name}. "
                           "Provide data_dir for local files.")

    return dataset


def preprocess_clue_example(
    example: Dict,
    task_config: CLUETaskConfig,
    tokenizer,
    max_length: int = 512,
) -> Dict:
    """
    Preprocess a single CLUE example.

    Args:
        example: Raw example dict
        task_config: Task configuration
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length

    Returns:
        Preprocessed example with input_ids, attention_mask, labels
    """
    # Get text(s)
    if len(task_config.text_columns) == 1:
        text = example[task_config.text_columns[0]]
        encoding = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
        )
    else:
        text_a = example[task_config.text_columns[0]]
        text_b = example[task_config.text_columns[1]]
        encoding = tokenizer(
            text_a,
            text_b,
            max_length=max_length,
            padding='max_length',
            truncation=True,
        )

    # Add label
    if task_config.label_column in example:
        encoding['labels'] = example[task_config.label_column]

    return encoding


def compute_clue_metrics(
    task_name: str,
    eval_pred: EvalPrediction,
) -> Dict[str, float]:
    """
    Compute metrics for a CLUE task.

    Args:
        task_name: Name of the CLUE task
        eval_pred: Evaluation predictions

    Returns:
        Dict of metric names to values
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = predictions.argmax(axis=-1)

    # Accuracy
    accuracy = (preds == labels).mean()

    metrics = {'accuracy': accuracy}

    # Task-specific metrics
    if task_name in ['afqmc', 'cmnli']:
        # F1 for pair tasks
        try:
            from sklearn.metrics import f1_score
            f1 = f1_score(labels, preds, average='macro')
            metrics['f1'] = f1
        except ImportError:
            pass

    return metrics


class CLUEEvaluator:
    """
    Evaluator for running CLUE benchmark tasks.

    Usage:
        evaluator = CLUEEvaluator(model, tokenizer, cnm_vocab)
        results = evaluator.evaluate_task('afqmc')
        all_results = evaluator.evaluate_all()
    """

    TASKS = ['afqmc', 'tnews', 'iflytek', 'cmnli', 'csl', 'wsc']

    def __init__(
        self,
        model,
        tokenizer,
        cnm_vocab=None,
        device: Optional[torch.device] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize CLUE evaluator.

        Args:
            model: CNM-BERT model (or model path)
            tokenizer: Tokenizer instance
            cnm_vocab: CNM vocabulary
            device: Device to use
            data_dir: Directory containing CLUE data files
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if isinstance(model, str):
            from cnm.model.modeling_cnm import CNMBertForSequenceClassification
            self.model = CNMBertForSequenceClassification.from_pretrained(model)
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.cnm_vocab = cnm_vocab
        self.data_dir = data_dir

    def evaluate_task(
        self,
        task_name: str,
        split: str = 'validation',
        batch_size: int = 32,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single CLUE task.

        Args:
            task_name: Name of the task
            split: Dataset split to evaluate
            batch_size: Evaluation batch size
            max_length: Maximum sequence length

        Returns:
            Dict of metric names to values
        """
        task_config = get_task_config(task_name)

        # Load dataset
        dataset = load_clue_dataset(task_name, split, self.data_dir)

        # Preprocess
        def preprocess_fn(example):
            return preprocess_clue_example(
                example, task_config, self.tokenizer, max_length
            )

        dataset = dataset.map(preprocess_fn, batched=False)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Evaluate
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')

                # Add structural inputs if available
                if self.cnm_vocab is not None and hasattr(self.tokenizer, 'prepare_structural_inputs'):
                    unique_trees, unique_to_position = self.tokenizer.prepare_structural_inputs(
                        batch['input_ids'], device=self.device
                    )
                    batch['unique_trees'] = unique_trees
                    batch['unique_to_position'] = unique_to_position

                outputs = self.model(**batch)
                preds = outputs.logits.argmax(dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        eval_pred = EvalPrediction(predictions=all_preds, label_ids=all_labels)
        metrics = compute_clue_metrics(task_name, eval_pred)

        return metrics

    def evaluate_all(
        self,
        split: str = 'validation',
        batch_size: int = 32,
        max_length: int = 512,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all CLUE tasks.

        Returns:
            Dict mapping task names to metrics
        """
        results = {}

        for task_name in self.TASKS:
            print(f"Evaluating {task_name}...")
            try:
                metrics = self.evaluate_task(
                    task_name, split, batch_size, max_length
                )
                results[task_name] = metrics
                print(f"  {task_name}: {metrics}")
            except Exception as e:
                print(f"  {task_name}: FAILED - {e}")
                results[task_name] = {'error': str(e)}

        # Compute average
        accuracies = [
            m['accuracy'] for m in results.values()
            if 'accuracy' in m
        ]
        if accuracies:
            results['average'] = {'accuracy': sum(accuracies) / len(accuracies)}

        return results

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
