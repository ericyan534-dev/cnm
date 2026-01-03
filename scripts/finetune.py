#!/usr/bin/env python3
"""
Fine-tune CNM-BERT on CLUE benchmark tasks.

Usage:
    python scripts/finetune.py \
        --model_path outputs/pretrain \
        --task_name afqmc \
        --output_dir outputs/finetune/afqmc
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from transformers import set_seed

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer
from cnm.data.collator import CNMDataCollatorForFineTuning
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import CNMBertForSequenceClassification
from cnm.training.args import CNMFinetuningArguments
from cnm.training.trainer import CNMTrainerForSequenceClassification, compute_metrics_for_classification
from cnm.evaluation.clue import (
    get_task_config,
    load_clue_dataset,
    preprocess_clue_example,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CNM-BERT on CLUE tasks')

    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained CNM-BERT model')
    parser.add_argument('--cnm_vocab_path', type=str,
                       help='Path to CNM vocabulary (auto-detected if not provided)')

    # Task arguments
    parser.add_argument('--task_name', type=str, required=True,
                       choices=['afqmc', 'tnews', 'iflytek', 'cmnli', 'csl', 'wsc'],
                       help='CLUE task name')
    parser.add_argument('--data_dir', type=str,
                       help='Local CLUE data directory (optional)')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32,
                       help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Performance arguments
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get task config
    task_config = get_task_config(args.task_name)
    logger.info(f"Task: {args.task_name} ({task_config.num_labels} labels)")

    # Load CNM vocabulary
    cnm_vocab_path = args.cnm_vocab_path
    if cnm_vocab_path is None:
        cnm_vocab_path = Path(args.model_path) / 'cnm_vocab.json'

    if Path(cnm_vocab_path).exists():
        logger.info(f"Loading CNM vocabulary from {cnm_vocab_path}")
        cnm_vocab = CNMVocab.load(Path(cnm_vocab_path))
    else:
        logger.warning("CNM vocabulary not found, structural embeddings disabled")
        cnm_vocab = None

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = CNMTokenizer.from_pretrained(
        args.model_path,
        cnm_vocab_file=str(cnm_vocab_path) if Path(cnm_vocab_path).exists() else None,
    )

    # Load model config and update for classification
    logger.info(f"Loading model from {args.model_path}")
    config = CNMConfig.from_pretrained(args.model_path)
    config.num_labels = task_config.num_labels

    # Load model
    model = CNMBertForSequenceClassification.from_pretrained(
        args.model_path,
        config=config,
    )

    # Load datasets
    logger.info("Loading CLUE datasets...")
    train_dataset = load_clue_dataset(args.task_name, 'train', args.data_dir)
    eval_dataset = load_clue_dataset(args.task_name, 'validation', args.data_dir)

    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Preprocess datasets
    def preprocess_fn(example):
        return preprocess_clue_example(
            example, task_config, tokenizer, args.max_seq_length
        )

    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(preprocess_fn, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess_fn, remove_columns=eval_dataset.column_names)

    # Data collator
    data_collator = CNMDataCollatorForFineTuning(
        tokenizer=tokenizer,
        cnm_vocab=cnm_vocab,
        max_length=args.max_seq_length,
    )

    # Training arguments
    training_args = CNMFinetuningArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="wandb",
        run_name=f"cnm-bert-{args.task_name}",
    )

    # Create trainer
    trainer = CNMTrainerForSequenceClassification(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_classification,
        cnm_vocab=cnm_vocab,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()

    # Evaluate
    logger.info("Final evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Results: {metrics}")

    # Save
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save results
    results_path = Path(args.output_dir) / 'results.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
