#!/usr/bin/env python3
"""
RLVR training script for chess puzzles using GRPO with Stockfish rewards.

Flow: SFT (learn format + basic chess) -> RLVR (refine move quality via Stockfish)

Uses Group Relative Policy Optimization (GRPO) which eliminates the need for a
value model by normalizing rewards within groups of completions per prompt.
"""

import argparse
import json
import os
import random
import sys
import time
import logging
from pathlib import Path

import chess
import chess.engine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
from huggingface_hub import login
from datasets import Dataset
import wandb

from util import (
    ChessPuzzle,
    PuzzleResponse,
    load_puzzle_data,
    compare_solutions,
    normalize_solution,
    find_stockfish,
    StockfishManager,
)
from config import MODEL_CONFIGS

logging.getLogger("transformers.training_args").setLevel(logging.ERROR)

RESPONSE_TEMPLATE = "JSON Output:\n"

# Global StockfishManager instance, initialized before training
_stockfish_manager: StockfishManager = None


def get_huggingface_api_key():
    """Read Hugging Face API key from environment variable or .huggingface_api_key file."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if api_key:
        return api_key.strip()
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


def build_prompt(fen: str) -> str:
    """Build the prompt string for a chess puzzle FEN."""
    return f"Analyze the following chess position and output the best sequence of moves: {fen}\n{RESPONSE_TEMPLATE}"


def build_dataset(puzzles: list[ChessPuzzle]) -> Dataset:
    """
    Build a dataset for GRPOTrainer.

    GRPOTrainer expects a "prompt" column. Extra columns (fen, expected_solution)
    are passed as **kwargs to reward functions.
    """
    records = []
    for puzzle in puzzles:
        records.append({
            "prompt": build_prompt(puzzle.fen),
            "fen": puzzle.fen,
            "expected_solution": puzzle.solution,
        })
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def format_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Reward for valid JSON format with a "solution" string field.
    +1.0 if valid, -1.0 if not.
    """
    rewards = []
    for completion in completions:
        try:
            parsed = json.loads(completion.strip())
            if isinstance(parsed, dict) and isinstance(parsed.get("solution"), str):
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        except (json.JSONDecodeError, ValueError):
            rewards.append(-1.0)
    return rewards


def stockfish_reward_fn(completions: list[str], fen: list[str], **kwargs) -> list[float]:
    """
    Reward based on Stockfish evaluation of the model's first move.

    Computes centipawn loss vs Stockfish's best move:
      reward = max(-1.0, 1.0 - cp_loss / 200.0)

    Illegal/unparseable moves get -1.0.
    """
    global _stockfish_manager
    rewards = []
    for completion, position_fen in zip(completions, fen):
        try:
            parsed = json.loads(completion.strip())
            solution_str = parsed.get("solution", "")
            moves = normalize_solution(solution_str)
            if not moves:
                rewards.append(-1.0)
                continue

            first_move_san = moves[0]
            board = chess.Board(position_fen)

            # Parse model's first move (SAN)
            try:
                model_move = board.parse_san(first_move_san)
            except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                rewards.append(-1.0)
                continue

            # Get Stockfish's evaluation of the best position
            best_eval = _stockfish_manager.evaluate_position(board)

            # Push model's move and evaluate from opponent's perspective, then negate
            board.push(model_move)
            model_eval_opp = _stockfish_manager.evaluate_position(board)
            model_eval = -model_eval_opp

            cp_loss = best_eval - model_eval
            reward = max(-1.0, min(1.0, 1.0 - cp_loss / 200.0))
            rewards.append(reward)

        except chess.engine.EngineTerminatedError:
            # Engine crashed — restart and penalize this sample
            try:
                _stockfish_manager.stop()
                _stockfish_manager.start()
            except Exception:
                pass
            rewards.append(-1.0)
        except Exception:
            rewards.append(-1.0)
    return rewards


def solution_match_reward_fn(completions: list[str], expected_solution: list[str], **kwargs) -> list[float]:
    """
    Partial sequence match reward against the known puzzle answer.
    Returns partial_score (0.0-1.0) from compare_solutions().
    """
    rewards = []
    for completion, expected in zip(completions, expected_solution):
        try:
            parsed = json.loads(completion.strip())
            solution_str = parsed.get("solution", "")
            comparison = compare_solutions(expected, solution_str)
            rewards.append(comparison.partial_score)
        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Training callback
# ---------------------------------------------------------------------------

class CustomRLCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 70)
        print("RLVR Training Begin (GRPO)")
        print(f"   Total Steps: {state.max_steps}")
        print("=" * 70 + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            elapsed = time.time() - self.start_time
            progress = state.global_step / state.max_steps
            eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0

            loss_str = f"Loss {logs['loss']:.4f}" if 'loss' in logs else ""
            reward_str = ""
            if 'reward' in logs:
                reward_str = f"Reward {logs['reward']:.3f}"
            elif 'rewards/mean' in logs:
                reward_str = f"Reward {logs['rewards/mean']:.3f}"

            parts = [
                f"[{progress * 100:>5.1f}%]",
                f"Step {state.global_step:>4}/{state.max_steps}",
            ]
            if loss_str:
                parts.append(loss_str)
            if reward_str:
                parts.append(reward_str)
            parts.append(f"ETA {int(eta // 3600):>1}h {int((eta % 3600) // 60):>2}m")
            print(" | ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.start_time
        print("\n" + "=" * 70)
        print(f"RLVR Training Complete! Time: {int(total // 60)}m {int(total % 60)}s")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Test rewards mode
# ---------------------------------------------------------------------------

def test_rewards(puzzles: list[ChessPuzzle], stockfish_manager: StockfishManager):
    """Run reward functions on a few puzzles with ground-truth and garbage completions."""
    global _stockfish_manager
    _stockfish_manager = stockfish_manager

    test_puzzles = puzzles[:min(3, len(puzzles))]
    print(f"\nTesting reward functions on {len(test_puzzles)} puzzles\n")

    for puzzle in test_puzzles:
        print(f"FEN: {puzzle.fen}")
        print(f"Expected: {puzzle.solution}")

        # Ground-truth completion
        good_completion = json.dumps({"solution": puzzle.solution})
        # Garbage completion
        bad_completion = '{"solution": "garbage moves Zz9"}'
        # Invalid JSON
        invalid_completion = "not json at all"

        completions = [good_completion, bad_completion, invalid_completion]
        labels = ["Ground truth", "Bad move", "Invalid JSON"]
        fens = [puzzle.fen] * 3
        expected = [puzzle.solution] * 3

        fmt_rewards = format_reward_fn(completions)
        sf_rewards = stockfish_reward_fn(completions, fen=fens)
        sol_rewards = solution_match_reward_fn(completions, expected_solution=expected)

        for label, fmt_r, sf_r, sol_r in zip(labels, fmt_rewards, sf_rewards, sol_rewards):
            print(f"  {label:15s}: format={fmt_r:+.1f}  stockfish={sf_r:+.3f}  solution={sol_r:.3f}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='RLVR training for chess puzzles using GRPO with Stockfish rewards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python train_rl.py --model-config llama --sft-adapter-dir models/my-sft-model --puzzle-file data/lichess-popular-250k.json --num-samples 5000',
    )
    parser.add_argument('--puzzle-file', type=str, default='data/wtharvey-sample.json',
                        help='Path to JSON puzzle file')
    parser.add_argument('--model-config', type=str, default='llama',
                        help='Model configuration (default: llama)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of puzzles to use (default: 1000, -1 for all)')
    parser.add_argument('--output-model-dir', type=str, default=None,
                        help='Output directory for the RL model')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--lora-r', type=int, default=64,
                        help='LoRA rank for RL phase (default: 64)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Per device batch size (default: 2 for CUDA, 1 for MPS/CPU)')
    parser.add_argument('--grad-steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=5e-7,
                        help='Learning rate (default: 5e-7)')

    # RL-specific arguments
    parser.add_argument('--sft-adapter-dir', type=str, default=None,
                        help='Path to SFT LoRA adapter directory to merge before RL training')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Total training steps (default: 500)')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='KL penalty coefficient (default: 0.01)')
    parser.add_argument('--num-generations', type=int, default=4,
                        help='Completions per prompt for GRPO (default: 4)')
    parser.add_argument('--max-completion-length', type=int, default=100,
                        help='Max tokens per completion (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default: 0.7)')
    parser.add_argument('--reward-weights', type=float, nargs=3, default=[0.2, 0.6, 0.2],
                        metavar=('FORMAT', 'STOCKFISH', 'SOLUTION'),
                        help='Weights for [format, stockfish, solution] rewards (default: 0.2 0.6 0.2)')

    # Stockfish arguments
    parser.add_argument('--stockfish-path', type=str, default=None,
                        help='Path to Stockfish binary (auto-detected if not provided)')
    parser.add_argument('--stockfish-depth', type=int, default=15,
                        help='Stockfish search depth (default: 15)')
    parser.add_argument('--stockfish-time-limit', type=float, default=0.1,
                        help='Stockfish time limit per evaluation in seconds (default: 0.1)')

    # Test mode
    parser.add_argument('--test-rewards', action='store_true',
                        help='Test reward functions on a few puzzles and exit')

    args = parser.parse_args()

    # Determine device
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

    # Validate model config
    if args.model_config not in MODEL_CONFIGS:
        print(f"Error: Unknown model config '{args.model_config}'. Available: {list(MODEL_CONFIGS.keys())}", file=sys.stderr)
        return 1
    model_config = MODEL_CONFIGS[args.model_config]
    base_model_name = model_config["name"]

    # Find and start Stockfish
    try:
        sf_path = find_stockfish(args.stockfish_path)
        print(f"Stockfish: {sf_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    stockfish_manager = StockfishManager(sf_path, depth=args.stockfish_depth, time_limit=args.stockfish_time_limit)

    # Load puzzles
    try:
        all_puzzles = load_puzzle_data(args.puzzle_file)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in puzzle file: {e}", file=sys.stderr)
        return 1

    if args.num_samples == -1:
        puzzles = all_puzzles
    else:
        num_samples = min(args.num_samples, len(all_puzzles))
        puzzles = random.sample(all_puzzles, num_samples)

    print(f"Loaded {len(puzzles)} puzzles from {Path(args.puzzle_file).name}")

    # Test rewards mode
    if args.test_rewards:
        with stockfish_manager:
            test_rewards(puzzles, stockfish_manager)
        return 0

    # Login to HuggingFace
    login(get_huggingface_api_key())

    # Model dtype and device map
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float16
    model_device_map = "auto" if device == "cuda" else device

    # Flash Attention on CUDA
    attn_impl = None
    if device == "cuda":
        try:
            import flash_attn  # type: ignore
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        except ImportError:
            print("Flash Attention 2 not available, using default attention")

    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=model_dtype,
        device_map=model_device_map,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    # Load tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge SFT adapters if provided
    merged_sft = False
    if args.sft_adapter_dir:
        print(f"Loading SFT LoRA adapters: {args.sft_adapter_dir}")
        model = PeftModel.from_pretrained(model, args.sft_adapter_dir)
        print("Merging SFT adapters into base model")
        model = model.merge_and_unload()
        # Remove residual peft metadata so GRPOTrainer doesn't warn about multiple adapters
        if hasattr(model, "peft_config"):
            del model.peft_config
        merged_sft = True

    # Fresh LoRA config for RL phase
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        target_modules=model_config["lora_targets"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Build dataset
    dataset = build_dataset(puzzles)
    print(f"Dataset: {len(dataset)} prompts")

    # Batch size defaults
    batch_size = args.batch_size if args.batch_size is not None else (2 if device == "cuda" else 1)

    # Output directory
    if not args.output_model_dir:
        suffix = "-sft" if merged_sft else ""
        output_model_dir = f"models/{base_model_name}-{len(puzzles)}-rl-steps-{args.max_steps}-lora-{args.lora_r}{suffix}"
    else:
        output_model_dir = args.output_model_dir

    # W&B
    if args.use_wandb:
        wandb.init(
            project="chess-puzzle-solver",
            name=f"{base_model_name}-{len(puzzles)}-rl-grpo",
            config={
                "model": base_model_name,
                "lora_r": args.lora_r,
                "learning_rate": args.learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": args.grad_steps,
                "dataset_size": len(puzzles),
                "max_steps": args.max_steps,
                "beta": args.beta,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "reward_weights": args.reward_weights,
                "sft_adapter_dir": args.sft_adapter_dir,
            },
        )

    # Build weighted reward function
    w_fmt, w_sf, w_sol = args.reward_weights

    def combined_reward_fn(completions: list[str], **kwargs) -> list[float]:
        fmt_rewards = format_reward_fn(completions, **kwargs)
        sf_rewards = stockfish_reward_fn(completions, **kwargs)
        sol_rewards = solution_match_reward_fn(completions, **kwargs)
        return [
            w_fmt * f + w_sf * s + w_sol * m
            for f, s, m in zip(fmt_rewards, sf_rewards, sol_rewards)
        ]

    # GRPOConfig defaults generation_batch_size to per_device_train_batch_size,
    # but it must be divisible by num_generations. Auto-fix if needed.
    generation_batch_size = batch_size
    if generation_batch_size % args.num_generations != 0:
        generation_batch_size = args.num_generations
        print(f"Warning: generation_batch_size ({batch_size}) not divisible by num_generations ({args.num_generations}), "
              f"setting generation_batch_size={generation_batch_size}")

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=output_model_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_steps,
        gradient_checkpointing=(device == "cuda"),
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        max_steps=args.max_steps,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="wandb" if args.use_wandb else "none",
        loss_type="grpo",
        dataloader_pin_memory=(device != "mps"),
        dataloader_num_workers=4 if device == "cuda" else 0,
    )

    # Print configuration
    print("\n" + "=" * 70)
    print("RLVR Training Configuration (GRPO)")
    print("=" * 70)
    print(f"  Puzzle file:           {args.puzzle_file}")
    print(f"  Model config:          {args.model_config} ({base_model_name})")
    print(f"  SFT adapter:           {args.sft_adapter_dir or 'None (training from base)'}")
    print(f"  Number of puzzles:     {len(puzzles)}")
    print(f"  Output model dir:      {output_model_dir}")
    print(f"  LoRA rank (r):         {args.lora_r}")
    print(f"  LoRA alpha:            {2 * args.lora_r}")
    print(f"  Batch size:            {batch_size}")
    print(f"  Gradient steps:        {args.grad_steps}")
    print(f"  Num generations:       {args.num_generations}")
    print(f"  Effective batch:       {batch_size * args.num_generations * args.grad_steps}")
    print(f"  Max steps:             {args.max_steps}")
    print(f"  Learning rate:         {args.learning_rate}")
    print(f"  Beta (KL penalty):     {args.beta}")
    print(f"  Temperature:           {args.temperature}")
    print(f"  Max completion len:    {args.max_completion_length}")
    print(f"  Reward weights:        format={w_fmt}, stockfish={w_sf}, solution={w_sol}")
    print(f"  Stockfish depth:       {args.stockfish_depth}")
    print(f"  Device:                {device}")
    print("=" * 70 + "\n")

    # Start Stockfish and train
    global _stockfish_manager
    with stockfish_manager:
        _stockfish_manager = stockfish_manager

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=combined_reward_fn,
            args=grpo_config,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            callbacks=[CustomRLCallback()],
        )
        trainer.remove_callback(PrinterCallback)

        trainer.train()
        trainer.save_model(output_model_dir)
        print(f"Model saved to: {output_model_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
