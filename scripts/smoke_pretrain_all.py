#!/usr/bin/env python3
"""
Comprehensive smoke test for CNM-BERT pretraining pipeline.

This script runs a complete smoke test of the training infrastructure:
- P0: Environment and GPU visibility check
- P1: CNMVocab load smoke
- P2: CNMTokenizer.from_pretrained + single encode smoke
- P3: Raw dataset load check (schema validation)
- P4: Tiny dataset → tokenize → collate → trainer.train(max_steps=2)

Additionally runs two multi-GPU subruns:
- Subrun A: DataParallel-like (non-torchrun, multiple GPUs visible)
- Subrun B: DDP via torchrun --nproc_per_node=8

Usage:
    python scripts/smoke_pretrain_all.py --train_file data/corpus --cnm_vocab_path data/ids/cnm_vocab.json

    # Quick mode (skip multi-GPU subruns, for debugging):
    python scripts/smoke_pretrain_all.py --train_file data/corpus --quick
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class PhaseResult:
    """Result of a single smoke phase."""
    phase: str
    passed: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_lines: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeTestReport:
    """Complete smoke test report."""
    phases: List[PhaseResult] = field(default_factory=list)
    subrun_a_result: Optional[Dict] = None  # DataParallel
    subrun_b_result: Optional[Dict] = None  # DDP


def run_phase(phase_name: str, fn, *args, **kwargs) -> PhaseResult:
    """Run a phase with exception handling."""
    try:
        details = fn(*args, **kwargs)
        return PhaseResult(
            phase=phase_name,
            passed=True,
            details=details or {}
        )
    except Exception as e:
        tb_lines = traceback.format_exc().split('\n')[-10:]
        return PhaseResult(
            phase=phase_name,
            passed=False,
            error_type=type(e).__name__,
            error_message=str(e)[:500],
            traceback_lines=tb_lines
        )


def phase_p0_environment() -> Dict[str, Any]:
    """P0: Environment and GPU visibility check."""
    import torch

    result = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'local_rank': os.environ.get('LOCAL_RANK', 'not_set'),
        'world_size': os.environ.get('WORLD_SIZE', 'not_set'),
    }

    if torch.cuda.is_available():
        result['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    print(f"  Python: {result['python_version'].split()[0]}")
    print(f"  PyTorch: {result['torch_version']}")
    print(f"  CUDA available: {result['cuda_available']}")
    print(f"  GPU count: {result['cuda_device_count']}")

    return result


def phase_p1_vocab_load(cnm_vocab_path: str) -> Dict[str, Any]:
    """P1: CNMVocab load smoke."""
    from cnm.data.vocab import CNMVocab

    vocab = CNMVocab.load(Path(cnm_vocab_path))

    result = {
        'component_vocab_size': vocab.component_vocab_size,
        'operator_vocab_size': vocab.operator_vocab_size,
        'num_indexed_chars': len(vocab.char_to_indexed),
    }

    print(f"  Components: {result['component_vocab_size']}")
    print(f"  Operators: {result['operator_vocab_size']}")
    print(f"  Indexed chars: {result['num_indexed_chars']}")

    return result


def phase_p2_tokenizer(pretrained_bert: str, cnm_vocab_path: str) -> Dict[str, Any]:
    """P2: CNMTokenizer.from_pretrained + single encode smoke."""
    from cnm.data.vocab import CNMVocab
    from cnm.data.tokenizer import CNMTokenizer

    cnm_vocab = CNMVocab.load(Path(cnm_vocab_path))
    tokenizer = CNMTokenizer.from_pretrained(pretrained_bert, cnm_vocab=cnm_vocab)

    # Test encoding
    test_text = "你好世界，这是一个测试。"
    encoding = tokenizer(test_text, return_tensors='pt')

    result = {
        'vocab_size': len(tokenizer),
        'test_input_ids_shape': list(encoding['input_ids'].shape),
        'has_struct_ids': 'struct_ids' in encoding,
    }

    print(f"  Tokenizer vocab size: {result['vocab_size']}")
    print(f"  Test encoding shape: {result['test_input_ids_shape']}")
    print(f"  Has struct_ids: {result['has_struct_ids']}")

    return result


def phase_p3_dataset_load(train_file: str) -> Dict[str, Any]:
    """P3: Raw dataset load check (schema validation)."""
    from datasets import load_dataset

    train_path = Path(train_file)

    if train_path.is_dir():
        data_files = list(train_path.glob('**/*.jsonl'))[:5]  # Only check first 5
        if not data_files:
            raise ValueError(f"No JSONL files found in {train_path}")
    else:
        data_files = [train_path]

    # Try loading just a few examples to validate schema
    schemas_seen = set()
    total_lines = 0

    for f in data_files:
        with open(f, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                if i >= 10:  # Only check first 10 lines per file
                    break
                try:
                    obj = json.loads(line.strip())
                    if isinstance(obj, dict):
                        schemas_seen.add(tuple(sorted(obj.keys())))
                        total_lines += 1
                except:
                    pass

    result = {
        'files_checked': len(data_files),
        'lines_sampled': total_lines,
        'unique_schemas': len(schemas_seen),
        'schemas': [list(s) for s in list(schemas_seen)[:5]],
    }

    print(f"  Files checked: {result['files_checked']}")
    print(f"  Lines sampled: {result['lines_sampled']}")
    print(f"  Unique schemas: {result['unique_schemas']}")

    return result


def create_tiny_sample(train_file: str, output_dir: str, max_examples: int = 200) -> str:
    """Create a tiny standardized sample JSONL for smoke testing."""
    train_path = Path(train_file)
    output_path = Path(output_dir) / 'smoke_sample.jsonl'

    examples = []

    if train_path.is_dir():
        data_files = list(train_path.glob('**/*.jsonl'))
    else:
        data_files = [train_path]

    for f in data_files:
        if len(examples) >= max_examples:
            break
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                if len(examples) >= max_examples:
                    break
                try:
                    obj = json.loads(line.strip())
                    if isinstance(obj, dict):
                        # Extract text
                        text = None
                        for key in ('text', 'content', 'body', 'article'):
                            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                                text = obj[key]
                                break

                        # Prepend title if present
                        if 'title' in obj and isinstance(obj['title'], str) and obj['title'].strip():
                            if text:
                                text = obj['title'] + '\n' + text
                            else:
                                text = obj['title']

                        if text and len(text.strip()) > 10:
                            examples.append({'text': text[:1000]})  # Truncate for speed
                except:
                    pass

    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    return str(output_path)


def phase_p4_trainer_smoke(
    sample_file: str,
    cnm_vocab_path: str,
    pretrained_bert: str,
    max_steps: int = 2,
    device: str = 'cuda:0'
) -> Dict[str, Any]:
    """P4: Tiny dataset → tokenize → collate → trainer.train(max_steps=2)."""
    import torch
    from datasets import load_dataset
    from transformers import BertModel

    from cnm.data.vocab import CNMVocab
    from cnm.data.tokenizer import CNMTokenizer
    from cnm.data.collator import CNMDataCollatorForPreTraining
    from cnm.model.configuration_cnm import CNMConfig
    from cnm.model.modeling_cnm import CNMBertForPreTraining
    from cnm.training.args import CNMPretrainingArguments
    from cnm.training.trainer import CNMTrainer

    # Load vocab and tokenizer
    cnm_vocab = CNMVocab.load(Path(cnm_vocab_path))
    tokenizer = CNMTokenizer.from_pretrained(pretrained_bert, cnm_vocab=cnm_vocab)

    # Create model config
    cnm_config = CNMConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,  # Small for speed
        num_attention_heads=2,
        intermediate_size=512,
        struct_dim=64,  # Small for speed
        tree_hidden_dim=128,
        max_tree_depth=6,
        component_vocab_size=cnm_vocab.component_vocab_size,
        operator_vocab_size=cnm_vocab.operator_vocab_size,
        aux_loss_weight=0.1,
    )

    # Create model
    model = CNMBertForPreTraining(cnm_config)

    # Load dataset
    dataset = load_dataset('json', data_files=sample_file, split='train')

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,  # Short for speed
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Split
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds = split['train']
    eval_ds = split['test']

    # Create collator
    collator = CNMDataCollatorForPreTraining(
        tokenizer=tokenizer,
        cnm_vocab=cnm_vocab,
        mlm_probability=0.15,
        wwm=False,  # Disable WWM for speed
        max_length=128,
    )

    # Create output dir
    output_dir = tempfile.mkdtemp(prefix='cnm_smoke_')

    # Training args
    training_args = CNMPretrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=max_steps,
        logging_steps=1,
        save_steps=1000,  # Don't save
        eval_steps=1000,  # Don't eval
        evaluation_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
    )

    # Create trainer
    trainer = CNMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        cnm_vocab=cnm_vocab,
    )

    # Train
    train_result = trainer.train()

    result = {
        'train_loss': train_result.training_loss,
        'global_step': trainer.state.global_step,
        'output_dir': output_dir,
    }

    print(f"  Train loss: {result['train_loss']:.4f}")
    print(f"  Steps completed: {result['global_step']}")

    return result


def run_subrun_dataparallel(
    sample_file: str,
    cnm_vocab_path: str,
    pretrained_bert: str,
    num_gpus: int = 8
) -> Dict[str, Any]:
    """
    Subrun A: DataParallel-like test.

    Runs training with multiple GPUs visible but no torchrun (DataParallel path).
    """
    script = f'''
import os
import sys
import json
import tempfile
sys.path.insert(0, "{Path(__file__).parent.parent / 'src'}")

import torch
from pathlib import Path
from datasets import load_dataset

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer
from cnm.data.collator import CNMDataCollatorForPreTraining
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import CNMBertForPreTraining
from cnm.training.args import CNMPretrainingArguments
from cnm.training.trainer import CNMTrainer

result = {{"passed": False, "error_type": None, "error_message": None}}

try:
    # Load components
    cnm_vocab = CNMVocab.load(Path("{cnm_vocab_path}"))
    tokenizer = CNMTokenizer.from_pretrained("{pretrained_bert}", cnm_vocab=cnm_vocab)

    # Small model for testing
    cnm_config = CNMConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        struct_dim=64,
        tree_hidden_dim=128,
        max_tree_depth=6,
        component_vocab_size=cnm_vocab.component_vocab_size,
        operator_vocab_size=cnm_vocab.operator_vocab_size,
        aux_loss_weight=0.1,
    )

    model = CNMBertForPreTraining(cnm_config)

    # Wrap with DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Load and tokenize data
    dataset = load_dataset('json', data_files="{sample_file}", split='train')

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, return_special_tokens_mask=True)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    collator = CNMDataCollatorForPreTraining(
        tokenizer=tokenizer, cnm_vocab=cnm_vocab,
        mlm_probability=0.15, wwm=False, max_length=128,
    )

    output_dir = tempfile.mkdtemp(prefix='cnm_dp_')

    training_args = CNMPretrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=2,
        logging_steps=1,
        evaluation_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = CNMTrainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        tokenizer=tokenizer,
        data_collator=collator,
        cnm_vocab=cnm_vocab,
    )

    trainer.train()
    result["passed"] = True
    result["details"] = {{"global_step": trainer.state.global_step, "gpus_used": torch.cuda.device_count()}}

except Exception as e:
    import traceback
    result["error_type"] = type(e).__name__
    result["error_message"] = str(e)[:500]
    result["traceback"] = traceback.format_exc().split("\\n")[-10:]

print("SMOKE_RESULT:" + json.dumps(result))
'''

    # Run in subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))

    try:
        proc = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        # Parse result
        for line in proc.stdout.split('\n'):
            if line.startswith('SMOKE_RESULT:'):
                return json.loads(line[len('SMOKE_RESULT:'):])

        return {
            'passed': False,
            'error_type': 'SubprocessError',
            'error_message': f'No result found. stdout: {proc.stdout[-500:]}, stderr: {proc.stderr[-500:]}'
        }

    except subprocess.TimeoutExpired:
        return {'passed': False, 'error_type': 'TimeoutError', 'error_message': 'Subrun timed out after 300s'}
    except Exception as e:
        return {'passed': False, 'error_type': type(e).__name__, 'error_message': str(e)}


def run_subrun_ddp(
    sample_file: str,
    cnm_vocab_path: str,
    pretrained_bert: str,
    num_gpus: int = 8
) -> Dict[str, Any]:
    """
    Subrun B: DDP test via torchrun.

    Runs training with torchrun --nproc_per_node=N to test DDP path.
    """
    # Create a temporary script file
    script_content = f'''#!/usr/bin/env python3
import os
import sys
import json
import tempfile
sys.path.insert(0, "{Path(__file__).parent.parent / 'src'}")

import torch
import torch.distributed as dist
from pathlib import Path
from datasets import load_dataset

from cnm.data.vocab import CNMVocab
from cnm.data.tokenizer import CNMTokenizer
from cnm.data.collator import CNMDataCollatorForPreTraining
from cnm.model.configuration_cnm import CNMConfig
from cnm.model.modeling_cnm import CNMBertForPreTraining
from cnm.training.args import CNMPretrainingArguments
from cnm.training.trainer import CNMTrainer
from transformers import set_seed

set_seed(42)

local_rank = int(os.environ.get('LOCAL_RANK', 0))
is_main = local_rank == 0

result = {{"passed": False, "error_type": None, "error_message": None}}

try:
    # Load components
    cnm_vocab = CNMVocab.load(Path("{cnm_vocab_path}"))
    tokenizer = CNMTokenizer.from_pretrained("{pretrained_bert}", cnm_vocab=cnm_vocab)

    # Small model
    cnm_config = CNMConfig(
        vocab_size=len(tokenizer),
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        struct_dim=64,
        tree_hidden_dim=128,
        max_tree_depth=6,
        component_vocab_size=cnm_vocab.component_vocab_size,
        operator_vocab_size=cnm_vocab.operator_vocab_size,
        aux_loss_weight=0.1,
    )

    model = CNMBertForPreTraining(cnm_config)

    # Load and tokenize data
    dataset = load_dataset('json', data_files="{sample_file}", split='train')

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, return_special_tokens_mask=True)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    collator = CNMDataCollatorForPreTraining(
        tokenizer=tokenizer, cnm_vocab=cnm_vocab,
        mlm_probability=0.15, wwm=False, max_length=128,
    )

    output_dir = tempfile.mkdtemp(prefix='cnm_ddp_')

    training_args = CNMPretrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=2,
        logging_steps=1,
        evaluation_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )

    trainer = CNMTrainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        tokenizer=tokenizer,
        data_collator=collator,
        cnm_vocab=cnm_vocab,
    )

    trainer.train()
    result["passed"] = True
    result["details"] = {{"global_step": trainer.state.global_step, "local_rank": local_rank}}

except Exception as e:
    import traceback
    result["error_type"] = type(e).__name__
    result["error_message"] = str(e)[:500]
    result["traceback"] = traceback.format_exc().split("\\n")[-10:]

if is_main:
    print("SMOKE_RESULT:" + json.dumps(result))
'''

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run with torchrun
        cmd = [
            'torchrun',
            f'--nproc_per_node={num_gpus}',
            script_path
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse result from rank 0
        for line in proc.stdout.split('\n'):
            if line.startswith('SMOKE_RESULT:'):
                return json.loads(line[len('SMOKE_RESULT:'):])

        return {
            'passed': False,
            'error_type': 'SubprocessError',
            'error_message': f'No result. stdout: {proc.stdout[-500:]}, stderr: {proc.stderr[-500:]}'
        }

    except subprocess.TimeoutExpired:
        return {'passed': False, 'error_type': 'TimeoutError', 'error_message': 'DDP subrun timed out'}
    except Exception as e:
        return {'passed': False, 'error_type': type(e).__name__, 'error_message': str(e)}
    finally:
        os.unlink(script_path)


def print_report(report: SmokeTestReport):
    """Print a formatted smoke test report."""
    print("\n" + "=" * 70)
    print("SMOKE TEST REPORT")
    print("=" * 70)

    # Phase results
    print("\n--- Phase Results ---")
    for phase in report.phases:
        status = "PASS" if phase.passed else "FAIL"
        print(f"  [{status}] {phase.phase}")
        if not phase.passed:
            print(f"        Error: {phase.error_type}: {phase.error_message}")

    # Subrun A (DataParallel)
    print("\n--- Subrun A: DataParallel ---")
    if report.subrun_a_result:
        status = "PASS" if report.subrun_a_result.get('passed') else "FAIL"
        print(f"  [{status}]")
        if not report.subrun_a_result.get('passed'):
            print(f"        Error: {report.subrun_a_result.get('error_type')}: {report.subrun_a_result.get('error_message')}")
    else:
        print("  [SKIPPED]")

    # Subrun B (DDP)
    print("\n--- Subrun B: DDP (torchrun) ---")
    if report.subrun_b_result:
        status = "PASS" if report.subrun_b_result.get('passed') else "FAIL"
        print(f"  [{status}]")
        if not report.subrun_b_result.get('passed'):
            print(f"        Error: {report.subrun_b_result.get('error_type')}: {report.subrun_b_result.get('error_message')}")
    else:
        print("  [SKIPPED]")

    # Summary
    print("\n--- Summary ---")
    phase_pass = sum(1 for p in report.phases if p.passed)
    phase_total = len(report.phases)
    print(f"  Phases: {phase_pass}/{phase_total} passed")

    all_passed = all(p.passed for p in report.phases)
    if report.subrun_a_result:
        all_passed = all_passed and report.subrun_a_result.get('passed', False)
    if report.subrun_b_result:
        all_passed = all_passed and report.subrun_b_result.get('passed', False)

    if all_passed:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED - see details above")

    print("=" * 70)

    # JSON output for programmatic parsing
    print("\n--- JSON Report ---")
    json_report = {
        'phases': [asdict(p) for p in report.phases],
        'subrun_a': report.subrun_a_result,
        'subrun_b': report.subrun_b_result,
        'all_passed': all_passed,
    }
    print(json.dumps(json_report, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Comprehensive smoke test for CNM-BERT')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Training data file or directory')
    parser.add_argument('--cnm_vocab_path', type=str, default='data/ids/cnm_vocab.json',
                        help='Path to CNM vocabulary')
    parser.add_argument('--pretrained_bert', type=str, default='bert-base-chinese',
                        help='Pretrained BERT model')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs for multi-GPU tests')
    parser.add_argument('--quick', action='store_true',
                        help='Skip multi-GPU subruns (for debugging)')
    parser.add_argument('--skip_subruns', action='store_true',
                        help='Skip multi-GPU subruns')

    args = parser.parse_args()

    report = SmokeTestReport()

    print("=" * 70)
    print("CNM-BERT SMOKE TEST")
    print("=" * 70)

    # P0: Environment
    print("\n[P0] Environment check...")
    report.phases.append(run_phase("P0: Environment", phase_p0_environment))

    # P1: Vocab load
    print("\n[P1] CNMVocab load...")
    report.phases.append(run_phase("P1: CNMVocab", phase_p1_vocab_load, args.cnm_vocab_path))

    # P2: Tokenizer
    print("\n[P2] CNMTokenizer load + encode...")
    report.phases.append(run_phase("P2: CNMTokenizer", phase_p2_tokenizer, args.pretrained_bert, args.cnm_vocab_path))

    # P3: Dataset load check
    print("\n[P3] Dataset schema check...")
    report.phases.append(run_phase("P3: Dataset", phase_p3_dataset_load, args.train_file))

    # Create tiny sample for P4 and subruns
    print("\n[Prep] Creating tiny sample dataset...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = create_tiny_sample(args.train_file, tmp_dir, max_examples=200)
        print(f"  Created: {sample_file}")

        # P4: Trainer smoke (single GPU)
        print("\n[P4] Trainer smoke test (single GPU)...")
        report.phases.append(run_phase(
            "P4: Trainer",
            phase_p4_trainer_smoke,
            sample_file,
            args.cnm_vocab_path,
            args.pretrained_bert,
            max_steps=2
        ))

        if not args.quick and not args.skip_subruns:
            # Subrun A: DataParallel
            print(f"\n[Subrun A] DataParallel test ({args.num_gpus} GPUs)...")
            report.subrun_a_result = run_subrun_dataparallel(
                sample_file,
                args.cnm_vocab_path,
                args.pretrained_bert,
                args.num_gpus
            )
            print(f"  Result: {'PASS' if report.subrun_a_result.get('passed') else 'FAIL'}")

            # Subrun B: DDP
            print(f"\n[Subrun B] DDP test ({args.num_gpus} GPUs via torchrun)...")
            report.subrun_b_result = run_subrun_ddp(
                sample_file,
                args.cnm_vocab_path,
                args.pretrained_bert,
                args.num_gpus
            )
            print(f"  Result: {'PASS' if report.subrun_b_result.get('passed') else 'FAIL'}")

    # Print report
    print_report(report)

    # Exit with appropriate code
    all_passed = all(p.passed for p in report.phases)
    if report.subrun_a_result:
        all_passed = all_passed and report.subrun_a_result.get('passed', False)
    if report.subrun_b_result:
        all_passed = all_passed and report.subrun_b_result.get('passed', False)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
