#!/usr/bin/env python3
"""
Pretrain CNM-BERT on Chinese corpus with MLM (Whole Word Masking) + auxiliary objectives.

This script implements the CNM-BERT pretraining as described in the paper:
- Masked Language Modeling with Whole Word Masking (WWM) for Chinese
- Auxiliary component prediction loss for structural retention

Usage:
    # Single GPU
    python scripts/pretrain.py \
        --train_file data/corpus \
        --cnm_vocab_path data/ids/cnm_vocab.json \
        --output_dir outputs/pretrain

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 scripts/pretrain.py \
        --train_file data/corpus \
        --cnm_vocab_path data/ids/cnm_vocab.json \
        --output_dir outputs/pretrain
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

import json
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import load_dataset
from transformers import (
    BertModel,
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
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description='Pretrain CNM-BERT with MLM + WWM')

    # Data arguments
    parser.add_argument('--train_file', type=str, required=True,
                       help='Training data file or directory containing JSONL files')
    parser.add_argument('--validation_file', type=str,
                       help='Validation data file (optional, will split from train if not provided)')
    parser.add_argument('--cnm_vocab_path', type=str, default='data/ids/cnm_vocab.json',
                       help='Path to CNM vocabulary JSON file')

    # Model arguments
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--pretrained_bert', type=str, default='bert-base-chinese',
                       help='Pretrained BERT model to initialize from (HuggingFace name or path)')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain',
                       help='Output directory for checkpoints and final model')
    parser.add_argument('--num_train_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32,
                       help='Training batch size per device')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64,
                       help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Peak learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # MLM arguments
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                       help='Probability of masking tokens')
    parser.add_argument('--wwm', action='store_true', default=True,
                       help='Use Whole Word Masking (default: True)')
    parser.add_argument('--no_wwm', action='store_true',
                       help='Disable Whole Word Masking')

    # Auxiliary loss
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                       help='Weight for auxiliary component prediction loss')

    # Performance arguments
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--bf16', action='store_true',
                       help='Use mixed precision training (BF16)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    # Logging arguments
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                       help='Evaluate every N steps')
    parser.add_argument('--save_total_limit', type=int, default=3,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--report_to', type=str, default='wandb',
                       choices=['wandb', 'tensorboard', 'none'],
                       help='Where to report metrics')
    parser.add_argument('--run_name', type=str, default='cnm-bert-pretrain',
                       help='Run name for logging')

    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Handle WWM flag
    if args.no_wwm:
        args.wwm = False

    # Load config if provided
    config_dict = load_config(Path(args.config))
    logger.info(f"Loaded config from {args.config}" if config_dict else "Using default config")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Determine local rank for distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_main_process = local_rank in [-1, 0]

    # Load CNM vocabulary
    if is_main_process:
        logger.info(f"Loading CNM vocabulary from {args.cnm_vocab_path}")
    cnm_vocab = CNMVocab.load(Path(args.cnm_vocab_path))
    if is_main_process:
        logger.info(f"Loaded vocabulary with {cnm_vocab.component_vocab_size} components, "
                   f"{cnm_vocab.operator_vocab_size} operators")

    # Load tokenizer using the fixed from_pretrained method
    if is_main_process:
        logger.info(f"Loading tokenizer from {args.pretrained_bert}")
    tokenizer = CNMTokenizer.from_pretrained(
        args.pretrained_bert,
        cnm_vocab=cnm_vocab,
    )
    if is_main_process:
        logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")

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
        aux_loss_weight=args.aux_loss_weight,
    )

    # Create model
    if is_main_process:
        logger.info("Creating CNM-BERT model...")
    model = CNMBertForPreTraining(cnm_config)

    # Load pretrained BERT weights
    if args.pretrained_bert:
        if is_main_process:
            logger.info(f"Loading pretrained weights from {args.pretrained_bert}")
        bert = BertModel.from_pretrained(args.pretrained_bert)
        model.bert._load_bert_weights(bert)
        del bert  # Free memory

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process:
            logger.info("Gradient checkpointing enabled")

    # Log model parameters
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load dataset
    if is_main_process:
        logger.info(f"Loading training data from {args.train_file}")
    train_path = Path(args.train_file)
    
    if train_path.is_dir():
        # Load from directory of JSONL files
        data_files = list(train_path.glob('**/*.jsonl'))
        if not data_files:
            raise ValueError(f"No JSONL files found in {train_path}")
        if is_main_process:
            logger.info(f"Found {len(data_files)} JSONL files")
            # ---- Debug: sniff first few lines of each file for keys (fast fail + visibility) ----
            def sniff_jsonl_keys(paths, max_lines=5):
                key_counter = Counter()
                bad_json = 0
                for p in paths:
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                if i >= max_lines:
                                    break
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict):
                                        key_counter[tuple(sorted(obj.keys()))] += 1
                                except Exception:
                                    bad_json += 1
                    except Exception:
                        bad_json += 1
                return key_counter, bad_json
        
            key_counter, bad_json = sniff_jsonl_keys([str(f) for f in data_files], max_lines=3)
            logger.info(f"Schema sniff: {len(key_counter)} unique key-sets; bad_json_lines={bad_json}")
            for ks, cnt in key_counter.most_common(10):
                logger.info(f"  keys={ks}  count={cnt}")

        # ---- Normalize to a single 'text' column (robust to title/text/content/body) ----
        def _to_text(examples):
            # examples is a dict of lists because batched=True
            n = len(next(iter(examples.values())))  # batch size
            out = []
            for i in range(n):
                t = None
        
                # Prefer existing text-like fields
                for key in ("text", "content", "body", "article", "main_text"):
                    if key in examples and examples[key] is not None:
                        v = examples[key][i]
                        if isinstance(v, str) and v.strip():
                            t = v
                            break
        
                # If we have a title, optionally prepend it
                title = None
                if "title" in examples and examples["title"] is not None:
                    v = examples["title"][i]
                    if isinstance(v, str) and v.strip():
                        title = v
        
                if t is None:
                    t = title or ""  # fallback to title only or empty
                elif title is not None and title not in t:
                    t = title + "\n" + t
        
                out.append(t)
        
            return {"text": out}

        from datasets import concatenate_datasets

        parts = []
        for f in data_files:
            ds_part = load_dataset(
                'json',
                data_files=str(f),
                split='train',
            )
            ds_part = ds_part.map(
                _to_text,
                batched=True,
                num_proc=1,
                desc="Normalizing text fields",
            )
            drop_cols = [c for c in ds_part.column_names if c != "text"]
            if drop_cols:
                ds_part = ds_part.remove_columns(drop_cols)
            ds_part = ds_part.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)
            parts.append(ds_part)

        dataset = concatenate_datasets(parts)

    else:
        dataset = load_dataset(
            'json',
            data_files=str(train_path),
            split='train',
        )

        # ---- Normalize to a single 'text' column (robust to title/text/content/body) ----
        def _to_text(examples):
            # examples is a dict of lists because batched=True
            n = len(next(iter(examples.values())))  # batch size
            out = []
            for i in range(n):
                t = None
        
                # Prefer existing text-like fields
                for key in ("text", "content", "body", "article", "main_text"):
                    if key in examples and examples[key] is not None:
                        v = examples[key][i]
                        if isinstance(v, str) and v.strip():
                            t = v
                            break
        
                # If we have a title, optionally prepend it
                title = None
                if "title" in examples and examples["title"] is not None:
                    v = examples["title"][i]
                    if isinstance(v, str) and v.strip():
                        title = v
        
                if t is None:
                    t = title or ""  # fallback to title only or empty
                elif title is not None and title not in t:
                    t = title + "\n" + t
        
                out.append(t)
        
            return {"text": out}
        
        dataset = dataset.map(
            _to_text,
            batched=True,
            num_proc=1,
            desc="Normalizing text fields",
        )
        
        # Keep only 'text' now that it exists
        drop_cols = [c for c in dataset.column_names if c != "text"]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
        
        # Filter empty rows to avoid wasting runtime
        dataset = dataset.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)
        
    if is_main_process:
        logger.info(f"Loaded {len(dataset)} training examples")
        if is_main_process:
            logger.info(f"After normalization: {len(dataset)} usable examples; columns={dataset.column_names}")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )

    if is_main_process:
        logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.dataloader_num_workers if local_rank == -1 else 1,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # Split for validation if no validation file
    if args.validation_file:
        val_dataset = load_dataset('json', data_files=args.validation_file, split='train')
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=args.dataloader_num_workers if local_rank == -1 else 1,
            remove_columns=val_dataset.column_names,
        )
    else:
        split = tokenized_dataset.train_test_split(test_size=0.01, seed=args.seed)
        tokenized_dataset = split['train']
        val_tokenized = split['test']
        if is_main_process:
            logger.info(f"Split: {len(tokenized_dataset)} train, {len(val_tokenized)} validation")

    # Create data collator with WWM
    if is_main_process:
        logger.info(f"Creating data collator (WWM={'enabled' if args.wwm else 'disabled'}, "
                   f"MLM prob={args.mlm_probability})")
    data_collator = CNMDataCollatorForPreTraining(
        tokenizer=tokenizer,
        cnm_vocab=cnm_vocab,
        mlm_probability=args.mlm_probability,
        wwm=args.wwm,
        max_length=args.max_seq_length,
    )

    # Training arguments
    training_args = CNMPretrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to if args.report_to != 'none' else [],
        run_name=args.run_name,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
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
    if is_main_process:
        logger.info("=" * 50)
        logger.info("Starting CNM-BERT Pretraining")
        logger.info("=" * 50)
        logger.info(f"  MLM Probability: {args.mlm_probability}")
        logger.info(f"  Whole Word Masking: {args.wwm}")
        logger.info(f"  Auxiliary Loss Weight: {args.aux_loss_weight}")
        logger.info(f"  Batch Size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps}")
        logger.info(f"  Learning Rate: {args.learning_rate}")
        logger.info(f"  Epochs: {args.num_train_epochs}")
        logger.info("=" * 50)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    if is_main_process:
        logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    if is_main_process:
        logger.info("Pretraining complete!")


if __name__ == '__main__':
    main()
