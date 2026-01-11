#!/usr/bin/env python3

import torch
import argparse
import time
import sys
import random
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login
from datasets import Dataset
import wandb
import logging

from util import ChessPuzzle, load_puzzle_data
from config import MODEL_CONFIGS

def get_huggingface_api_key():
    """
    Read Hugging Face API key from environment variable or .huggingface_api_key file.

    Returns:
        str: The API key

    Raises:
        FileNotFoundError: If no API key found in environment or file
        ValueError: If the API key is empty
    """
    # Check environment variables first (preferred for remote deployments)
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if api_key:
        return api_key.strip()

    # Fall back to file-based approach for local development
    api_key_file = Path('.huggingface_api_key')
    if not api_key_file.exists():
        raise FileNotFoundError(
            "No Hugging Face API key found. Either set HF_TOKEN environment variable "
            "or create a .huggingface_api_key file with your API key."
        )

    api_key = api_key_file.read_text().strip()
    if not api_key:
        raise ValueError("API key file (.huggingface_api_key) is empty")

    return api_key

def training_example(puzzle: ChessPuzzle, tokenizer: AutoTokenizer) -> str:
    """
    Create a training example from a puzzle.
    """
    prompt = f"""Analyze the following chess position and output the best sequence of moves: {puzzle.fen}
JSON Output:
{{ "solution": "{puzzle.solution}" }}{tokenizer.eos_token}"""
    return {"text": prompt}

class CustomTrainerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*70)
        print("🎯 Training Begin")
        print(f"   Dataset: {len(kwargs.get('train_dataloader', []))} batches")
        print(f"   Total Steps: {state.max_steps}")
        print("="*70 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            elapsed = time.time() - self.start_time
            progress = state.global_step / state.max_steps
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0
            steps = state.global_step
            tokens_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps * 512  # avg seq length
            total_tokens = steps * tokens_per_step
            tok_per_sec = total_tokens / elapsed

            print(
                f"[{progress*100:>5.1f}%] "
                f"Step {state.global_step:>4}/{state.max_steps} | "
                f"Loss {logs['loss']:.4f} | "
                f"Accuracy {logs.get('mean_token_accuracy', 0)*100:>5.1f}% | "
                f"LR {logs['learning_rate']:.2e} | "
                f"ETA {int(eta//3600):>1}h {int((eta%3600)//60):>2}m | "
                f"Speed {tok_per_sec:>8.1f} tokens/sec"
            )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(metrics.keys())
            print(
                f"  📊 Evaluation @ Step {state.global_step} | "
                f"Eval Loss: {metrics.get('eval_loss', 0):.4f} | "
                f"Accuracy: {metrics.get('eval_accuracy', 0)*100:>5.1f}%"
            )

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.start_time
        print("\n" + "="*70)
        print(f"✅ Training Complete! Time: {int(total//60)}m {int(total%60)}s")
        print("="*70 + "\n")

logging.getLogger("transformers.training_args").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model on chess puzzles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python train.py --puzzle_file data/wtharvey-sample.json --model-name meta-llama/Llama-3.2-3B'
    )
    parser.add_argument(
        '--puzzle-file',
        type=str,
        default='data/wtharvey-sample.json',
        help='Path to JSON puzzle file (e.g., data/wtharvey-sample.json)'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='llama',
        help='Model configuration (default: llama)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of problems to use (default: 100). This will be split into a training and test set.'
    )
    parser.add_argument(
        '--output-model-dir',
        type=str,
        help='Output directory for the fine-tuned model (default: models/<model_name>-<num_samples>-lora-<lora_r>)'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging for experiment tracking'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=32,
        help='LoRA rank (r) parameter (default: 32). lora_alpha will be set to 2 * lora_r.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Per device train batch size (default: 8 for CUDA, 2 for MPS/CPU)'
    )
    parser.add_argument(
        '--grad-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps (default: 4)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"Using device: {device} ({gpu_count}x {gpu_name})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Using device: {device}")
    else:
        device = "cpu"
        print(f"Using device: {device}")

    model_config = MODEL_CONFIGS[args.model_config]
    print(f"Loading model: {model_config['name']}")
    
    login(get_huggingface_api_key())
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], 
        device_map=device, 
        low_cpu_mem_usage=True, 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_puzzles = load_puzzle_data(args.puzzle_file)
    if args.num_samples == -1:
        puzzles = all_puzzles
    else:
        num_samples = min(args.num_samples, len(all_puzzles))
        puzzles = random.sample(all_puzzles, num_samples)
    puzzle_examples = [ training_example(puzzle, tokenizer) for puzzle in puzzles ]
    puzzles_dataset = Dataset.from_list(puzzle_examples).train_test_split(test_size=0.1)

    # Use bfloat16 on CUDA (better for Ampere+ GPUs), float16 on MPS
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float16
    # Use "auto" device_map on CUDA for multi-GPU support, specific device otherwise
    model_device_map = "auto" if device == "cuda" else device

    # Use Flash Attention 2 if available on CUDA (faster training)
    attn_impl = None
    if device == "cuda":
        try:
            import flash_attn # type: ignore
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention 2 not available, using default attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=model_dtype,
        device_map=model_device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,  # Typically set to 2x the rank
        target_modules=model_config["lora_targets"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Use larger batch size on CUDA (more VRAM available) if not specified
    batch_size = args.batch_size if args.batch_size is not None else (8 if device == "cuda" else 2)

    if args.use_wandb:
        wandb.init(
            project="chess-puzzle-solver",
            name=f"{model_config['name']}-{len(puzzles)}-lora-{args.lora_r}",
            config={
                "model": model_config["name"],
                "lora_r": args.lora_r,
                "lora_alpha": 2 * args.lora_r,
                "learning_rate": 2e-4,
                "batch_size": batch_size,
                "gradient_accumulation_steps": args.grad_steps,
                "dataset_size": len(puzzles),
                "epochs": args.epochs,
            }
        )

    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_steps,
        gradient_checkpointing=(device == "cuda"),
        bf16=(device == "cuda"),  # Use bfloat16 on CUDA
        fp16=(device == "mps"),   # Use float16 on MPS
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        warmup_steps=100,  # ~3% of training steps
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        report_to="wandb" if args.use_wandb else "none",
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_pin_memory=(device != "mps"),  # MPS doesn't support pin_memory
        dataloader_num_workers=4 if device == "cuda" else 0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=puzzles_dataset["train"],
        eval_dataset=puzzles_dataset["test"],
        args=training_args,
        processing_class=tokenizer,
        callbacks=[CustomTrainerCallback()],
    )
    trainer.remove_callback(PrinterCallback)

    trainer.train()
    if not args.output_model_dir:
        output_model_dir = f"models/{model_config['name']}-{len(puzzles)}-lora-{args.lora_r}"
    else:
        output_model_dir = args.output_model_dir
    trainer.save_model(output_model_dir)
    print(f"Model saved to: {output_model_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())