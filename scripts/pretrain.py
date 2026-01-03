#!/usr/bin/env python3
"""
Pretrain CNM-BERT on Chinese corpus with MLM + auxiliary objectives.

Usage:
    python scripts/pretrain.py \
        --train_file data/corpus \
        --output_dir outputs/pretrain \
        --config configs/pretrain.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    set_seed,
)

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer
from cnm.data.collator import CNMDataCollatorForPreTraining
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import CNMBertForPreTraining
from cnm.training.args import CNMPretrainingArguments
from cnm.training.trainer import CNMTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Pretrain CNM-BERT')

    # Data arguments
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training data file or directory')
    parser.add_argument('--validation_file', type=str,
                       help='Validation data file')
    parser.add_argument('--cnm_vocab_path', type=str, default='data/ids/cnm_vocab.json',
                       help='Path to CNM vocabulary')
    parser.add_argument('--ids_cache_path', type=str, default='data/ids/ids_cache.json',
                       help='Path to IDS parse cache')

    # Model arguments
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--pretrained_bert', type=str, default='bert-base-chinese',
                       help='Pretrained BERT model to initialize from')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain',
                       help='Output directory')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32,
                       help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Performance arguments
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    args = parser.parse_args()

    # Load config if provided
    config_dict = {}
    if args.config and Path(args.config).exists():
        config_dict = load_config(Path(args.config))
        logger.info(f"Loaded config from {args.config}")

    # Set seed
    set_seed(args.seed)

    # Load CNM vocabulary
    logger.info(f"Loading CNM vocabulary from {args.cnm_vocab_path}")
    cnm_vocab = CNMVocab.load(Path(args.cnm_vocab_path))
    logger.info(f"Loaded vocabulary with {cnm_vocab.component_vocab_size} components")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.pretrained_bert}")
    base_tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_bert)

    # Create CNM tokenizer
    tokenizer = CNMTokenizer(
        vocab_file=None,
        cnm_vocab=cnm_vocab,
    )
    tokenizer._tokenizer = base_tokenizer._tokenizer
    tokenizer._build_token_char_map()

    # Create model config
    model_config = config_dict.get('model', {})
    cnm_config = CNMConfig(
        vocab_size=len(tokenizer),
        hidden_size=model_config.get('hidden_size', 768),
        num_hidden_layers=model_config.get('num_hidden_layers', 12),
        num_attention_heads=model_config.get('num_attention_heads', 12),
        intermediate_size=model_config.get('intermediate_size', 3072),
        struct_dim=model_config.get('struct_dim', 256),
        tree_hidden_dim=model_config.get('tree_hidden_dim', 512),
        max_tree_depth=model_config.get('max_tree_depth', 6),
        component_vocab_size=cnm_vocab.component_vocab_size,
        operator_vocab_size=cnm_vocab.operator_vocab_size,
        aux_loss_weight=model_config.get('aux_loss_weight', 0.1),
    )

    # Create model
    logger.info("Creating CNM-BERT model...")
    model = CNMBertForPreTraining(cnm_config)

    # Load pretrained BERT weights
    if args.pretrained_bert:
        logger.info(f"Loading pretrained weights from {args.pretrained_bert}")
        model.bert._load_bert_weights(
            __import__('transformers').BertModel.from_pretrained(args.pretrained_bert)
        )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    logger.info(f"Loading training data from {args.train_file}")
    train_path = Path(args.train_file)

    if train_path.is_dir():
        # Load from directory of JSONL files
        data_files = list(train_path.glob('**/*.jsonl'))
        if not data_files:
            raise ValueError(f"No JSONL files found in {train_path}")
        dataset = load_dataset('json', data_files=[str(f) for f in data_files], split='train')
    else:
        dataset = load_dataset('json', data_files=str(train_path), split='train')

    logger.info(f"Loaded {len(dataset)} training examples")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            return_special_tokens_mask=True,
        )

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.dataloader_num_workers,
        remove_columns=['text'],
        desc="Tokenizing",
    )

    # Split for validation if no validation file
    if args.validation_file:
        val_dataset = load_dataset('json', data_files=args.validation_file, split='train')
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.dataloader_num_workers,
            remove_columns=['text'],
        )
    else:
        split = tokenized_dataset.train_test_split(test_size=0.05, seed=args.seed)
        tokenized_dataset = split['train']
        val_tokenized = split['test']
        logger.info(f"Split: {len(tokenized_dataset)} train, {len(val_tokenized)} validation")

    # Create data collator
    data_collator = CNMDataCollatorForPreTraining(
        tokenizer=tokenizer,
        cnm_vocab=cnm_vocab,
        mlm_probability=config_dict.get('training', {}).get('mlm_probability', 0.15),
        wwm=config_dict.get('training', {}).get('wwm', True),
    )

    # Training arguments
    training_config = config_dict.get('training', {})
    training_args = CNMPretrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 64),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        learning_rate=args.learning_rate,
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        fp16=args.fp16,
        logging_steps=training_config.get('logging_steps', 100),
        save_steps=training_config.get('save_steps', 1000),
        eval_steps=training_config.get('eval_steps', 1000),
        evaluation_strategy="steps",
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=training_config.get('report_to', 'wandb'),
        run_name=training_config.get('run_name', 'cnm-bert-pretrain'),
    )

    # Create trainer
    trainer = CNMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        cnm_vocab=cnm_vocab,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Done!")


if __name__ == '__main__':
    main()
