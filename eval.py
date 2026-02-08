#!/usr/bin/env python3
"""
Evaluate responses for chess puzzles using a language model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from util import (
    full_puzzle_prompt, ChessPuzzle, load_puzzle_data, PuzzleResponse,
    compare_solutions, normalize_solution, find_stockfish, StockfishManager,
)
from config import MODEL_CONFIGS

RESPONSE_TEMPLATE = "JSON Output:\n"

def eval_prompt(puzzle: ChessPuzzle) -> str:
    return f"""Analyze the following chess position and output the best sequence of moves: {puzzle.fen}
{RESPONSE_TEMPLATE}"""

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate responses for chess puzzles using a language model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python eval.py --puzzle_file data/wtharvey-sample.json --model-config llama --num-problems 10'
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
        help='Model configuration (default: llama). Options: llama, qwen'
    )
    parser.add_argument(
        '--trained-model-dir',
        type=str,
        default=None,
        help='Trained model directory (e.g., models/Llama-3.2-3B-10000-lora-64)'
    )
    parser.add_argument(
        '--sft-adapter-dir',
        type=str,
        default=None,
        help='SFT LoRA adapter directory to merge before loading RL adapters (required when evaluating RL models)'
    )
    parser.add_argument(
        '--num-problems',
        type=int,
        default=1,
        help='Number of problems to generate responses for (default: 1, use -1 for all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save results (default: prints to stdout)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Number of puzzles to process in each batch (default: 8)'
    )
    parser.add_argument(
        '--stockfish-path',
        type=str,
        default=None,
        help='Path to Stockfish binary (auto-detected if not provided). Enables Stockfish move quality scoring.'
    )
    parser.add_argument(
        '--stockfish-depth',
        type=int,
        default=15,
        help='Stockfish search depth for move evaluation (default: 15)'
    )
    args = parser.parse_args()
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.backends.mps.is_available():
        print(f"Emptying MPS cache")
        torch.mps.empty_cache()
    
    # Try to initialize Stockfish
    stockfish_manager = None
    try:
        sf_path = find_stockfish(args.stockfish_path)
        stockfish_manager = StockfishManager(sf_path, depth=args.stockfish_depth)
        stockfish_manager.start()
        print(f"Stockfish: {sf_path} (depth={args.stockfish_depth})")
    except FileNotFoundError:
        if args.stockfish_path:
            print(f"Warning: Stockfish not found at {args.stockfish_path}", file=sys.stderr)
        else:
            print("Note: Stockfish not found, skipping Stockfish move quality scoring.")

    # Get model configuration
    if args.model_config not in MODEL_CONFIGS:
        print(f"Error: Unknown model config '{args.model_config}'. Available: {list(MODEL_CONFIGS.keys())}", file=sys.stderr)
        return 1
    model_config = MODEL_CONFIGS[args.model_config]
    base_model_name = model_config["name"]
    
    # Use bfloat16 on CUDA (matches training), float16 on MPS/CPU
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float16
    # Use "auto" device_map on CUDA for multi-GPU support, specific device otherwise
    model_device_map = "auto" if device == "cuda" else device

    # Load model and tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="left",  # Left-padding is standard for generation
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=model_dtype,
        device_map=model_device_map,
        low_cpu_mem_usage=True,
    )
    # Merge SFT adapters first if provided (required for evaluating RL models)
    if args.sft_adapter_dir:
        print(f"Loading SFT LoRA adapters: {args.sft_adapter_dir}")
        model = PeftModel.from_pretrained(model, args.sft_adapter_dir, low_cpu_mem_usage=False)
        print("Merging SFT adapters into base model")
        model = model.merge_and_unload()
        if hasattr(model, "peft_config"):
            del model.peft_config

    if args.trained_model_dir:
        print(f"Loading LoRA adapters: {args.trained_model_dir}")
        model = PeftModel.from_pretrained(
            model,
            args.trained_model_dir,
            low_cpu_mem_usage=False,
        )
    else:
        print(f"No LoRA adapters found, using base model")

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    try:
        puzzles = load_puzzle_data(args.puzzle_file)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in puzzle file: {e}", file=sys.stderr)
        return 1
    
    # Limit number of problems
    if args.num_problems == -1:
        num_problems = len(puzzles)
    else:
        num_problems = min(args.num_problems, len(puzzles))
    puzzles_to_process = puzzles[:num_problems]
    
    print(f"Processing {num_problems} puzzles from {Path(args.puzzle_file).name} in batches of {args.batch_size}")

    # Generate responses in batches
    start_time = time.perf_counter()
    all_generated_texts = []

    for batch_start in range(0, len(puzzles_to_process), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(puzzles_to_process))
        batch_puzzles = puzzles_to_process[batch_start:batch_end]

        print(f"Processing batch {batch_start // args.batch_size + 1}/{(len(puzzles_to_process) + args.batch_size - 1) // args.batch_size} ({len(batch_puzzles)} puzzles)")

        prompt_batch = [eval_prompt(puzzle) for puzzle in batch_puzzles]
        inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs,
                max_new_tokens=100,  # JSON response is short
                do_sample=False,  # Greedy decoding for deterministic output
                pad_token_id=tokenizer.eos_token_id)
        generated_text_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_generated_texts.extend(generated_text_batch)

        # Clear cache after each batch to free memory
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    elapsed_time = time.perf_counter() - start_time

    # Process and display results
    results = []
    all_prompts = [eval_prompt(puzzle) for puzzle in puzzles_to_process]

    for puzzle, prompt, output in zip(puzzles_to_process, all_prompts, all_generated_texts):
        output_truncated = output.replace(prompt, '').strip()
        
        print(f"\nPuzzle: {puzzle.description}")
        print(f"FEN: {puzzle.fen}")
        print(f"Expected solution: {puzzle.solution}")
        
        try:
            model_response = PuzzleResponse.model_validate(json.loads(output_truncated))
            comparison = compare_solutions(puzzle.solution, model_response.solution)

            print(f"Model solution: {model_response.solution}")
            if comparison.exact_match:
                print(f"Result: EXACT MATCH")
            elif comparison.normalized_match:
                print(f"Result: NORMALIZED MATCH (move numbers differ)")
            elif comparison.first_move_correct:
                print(f"Result: PARTIAL ({comparison.correct_moves}/{comparison.total_moves} moves, {comparison.partial_score:.0%})")
            else:
                print(f"Result: INCORRECT (0/{comparison.total_moves} moves)")

            # Stockfish evaluation of model's first move
            stockfish_cp_loss = None
            illegal_move = False
            if stockfish_manager and comparison.model_moves:
                try:
                    board = chess.Board(puzzle.fen)
                    first_move_san = comparison.model_moves[0]
                    model_move = board.parse_san(first_move_san)
                    best_eval = stockfish_manager.evaluate_position(board)
                    board.push(model_move)
                    model_eval = -stockfish_manager.evaluate_position(board)
                    stockfish_cp_loss = best_eval - model_eval
                    print(f"Stockfish cp loss: {stockfish_cp_loss}")
                except (chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError):
                    illegal_move = True
                    print(f"Illegal move: {comparison.model_moves[0]}")
                except Exception:
                    stockfish_cp_loss = None

            result = {
                "fen": puzzle.fen,
                "description": puzzle.description,
                "citation": puzzle.citation,
                "expected_solution": puzzle.solution,
                "model_solution": model_response.solution,
                "exact_match": comparison.exact_match,
                "normalized_match": comparison.normalized_match,
                "first_move_correct": comparison.first_move_correct,
                "correct_moves": comparison.correct_moves,
                "total_moves": comparison.total_moves,
                "partial_score": comparison.partial_score,
                "stockfish_cp_loss": stockfish_cp_loss,
                "illegal_move": illegal_move,
            }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"Result: PARSE ERROR - {e}")
            print(f"Output: {output_truncated[:100]}")
            result = {
                "fen": puzzle.fen,
                "description": puzzle.description,
                "citation": puzzle.citation,
                "expected_solution": puzzle.solution,
                "error": str(e),
                "raw_output": output_truncated[:200]
            }
        
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    valid_results = [r for r in results if 'error' not in r]
    num_errors = len(results) - len(valid_results)

    num_exact = sum(1 for r in valid_results if r.get('exact_match', False))
    num_normalized = sum(1 for r in valid_results if r.get('normalized_match', False))
    num_first_move = sum(1 for r in valid_results if r.get('first_move_correct', False))
    total_partial_score = sum(r.get('partial_score', 0) for r in valid_results)

    num_illegal = sum(1 for r in valid_results if r.get('illegal_move', False))

    print(f"Total puzzles: {len(results)}")
    print(f"Parse errors: {num_errors}")
    print(f"Illegal moves: {num_illegal}")
    print(f"")
    print(f"Exact matches: {num_exact}")
    print(f"Normalized matches: {num_normalized} (correct moves, different formatting)")
    print(f"First move correct: {num_first_move}")
    print(f"")
    if len(results) > 0:
        print(f"Exact accuracy: {num_exact / len(results):.2%}")
        print(f"Normalized accuracy: {num_normalized / len(results):.2%}")
        print(f"First-move accuracy: {num_first_move / len(results):.2%}")
    if len(valid_results) > 0:
        print(f"Average partial score: {total_partial_score / len(valid_results):.2%}")

    # Stockfish summary
    sf_losses = [r['stockfish_cp_loss'] for r in valid_results if r.get('stockfish_cp_loss') is not None]
    if sf_losses:
        avg_cp_loss = sum(sf_losses) / len(sf_losses)
        optimal_count = sum(1 for cp in sf_losses if cp == 0)
        print(f"")
        print(f"Average Stockfish cp loss: {avg_cp_loss:.1f} (lower is better, {len(sf_losses)} evaluated)")
        print(f"Optimal move rate: {optimal_count / len(sf_losses):.2%} ({optimal_count}/{len(sf_losses)})")

    print(f"Total generation time: {elapsed_time:.2f} seconds")
    print(f"Average time per puzzle: {elapsed_time / num_problems:.2f} seconds")

    # Build run config and stats for saving
    run_config = {
        "puzzle_file": args.puzzle_file,
        "model_config": args.model_config,
        "base_model_name": base_model_name,
        "sft_adapter_dir": args.sft_adapter_dir,
        "trained_model_dir": args.trained_model_dir,
        "num_problems": num_problems,
        "batch_size": args.batch_size,
        "device": device,
    }
    stats = {
        "total_puzzles": len(results),
        "parse_errors": num_errors,
        "illegal_moves": num_illegal,
        "exact_matches": num_exact,
        "normalized_matches": num_normalized,
        "first_move_correct": num_first_move,
        "exact_accuracy": num_exact / len(results) if results else 0,
        "normalized_accuracy": num_normalized / len(results) if results else 0,
        "first_move_accuracy": num_first_move / len(results) if results else 0,
        "average_partial_score": total_partial_score / len(valid_results) if valid_results else 0,
        "total_generation_time_seconds": elapsed_time,
        "average_time_per_puzzle_seconds": elapsed_time / num_problems if num_problems else 0,
    }
    if sf_losses:
        stats["average_stockfish_cp_loss"] = sum(sf_losses) / len(sf_losses)
        stats["optimal_move_rate"] = sum(1 for cp in sf_losses if cp == 0) / len(sf_losses)
        stats["stockfish_evaluated_count"] = len(sf_losses)

    # Save results if output file specified
    if args.output:
        output_file = Path(args.output)
        output_data = {
            "config": run_config,
            "stats": stats,
            "results": results,
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    # Cleanup Stockfish
    if stockfish_manager:
        stockfish_manager.stop()

    return 0


if __name__ == '__main__':
    sys.exit(main())

