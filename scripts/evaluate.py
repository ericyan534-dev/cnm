#!/usr/bin/env python3
"""
Evaluate CNM-BERT on CLUE benchmark tasks.

Usage:
    python scripts/evaluate.py \
        --model_path outputs/finetune/afqmc \
        --task_name afqmc

    # Evaluate on all tasks
    python scripts/evaluate.py \
        --model_path outputs/pretrain \
        --all_tasks \
        --output_file results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer
from cnm.evaluation.clue import CLUEEvaluator, CLUE_TASKS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNM-BERT on CLUE')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned CNM-BERT model')
    parser.add_argument('--task_name', type=str,
                       choices=list(CLUE_TASKS.keys()),
                       help='CLUE task to evaluate')
    parser.add_argument('--all_tasks', action='store_true',
                       help='Evaluate on all CLUE tasks')
    parser.add_argument('--data_dir', type=str,
                       help='Local CLUE data directory')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['validation', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Evaluation batch size')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--output_file', type=str,
                       help='Output file for results (JSON)')
    parser.add_argument('--cnm_vocab_path', type=str,
                       help='Path to CNM vocabulary')

    args = parser.parse_args()

    if not args.task_name and not args.all_tasks:
        parser.error("Either --task_name or --all_tasks must be specified")

    # Load CNM vocabulary
    cnm_vocab_path = args.cnm_vocab_path
    if cnm_vocab_path is None:
        cnm_vocab_path = Path(args.model_path) / 'cnm_vocab.json'

    if Path(cnm_vocab_path).exists():
        logger.info(f"Loading CNM vocabulary from {cnm_vocab_path}")
        cnm_vocab = CNMVocab.load(Path(cnm_vocab_path))
    else:
        logger.warning("CNM vocabulary not found")
        cnm_vocab = None

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = CNMTokenizer.from_pretrained(
        args.model_path,
        cnm_vocab_file=str(cnm_vocab_path) if Path(cnm_vocab_path).exists() else None,
    )

    # Create evaluator
    logger.info(f"Loading model from {args.model_path}")
    evaluator = CLUEEvaluator(
        model=args.model_path,
        tokenizer=tokenizer,
        cnm_vocab=cnm_vocab,
        data_dir=args.data_dir,
    )

    # Evaluate
    if args.all_tasks:
        logger.info("Evaluating on all CLUE tasks...")
        results = evaluator.evaluate_all(
            split=args.split,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    else:
        logger.info(f"Evaluating on {args.task_name}...")
        task_results = evaluator.evaluate_task(
            args.task_name,
            split=args.split,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        results = {args.task_name: task_results}

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for task, metrics in results.items():
        if task == 'average':
            continue
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    if 'average' in results:
        print(f"\nAverage accuracy: {results['average']['accuracy']:.4f}")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

    logger.info("Done!")


if __name__ == '__main__':
    main()
