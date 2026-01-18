#!/usr/bin/env python3
"""
Generate responses for chess puzzles using a language model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from util import full_puzzle_prompt, ChessPuzzle, load_puzzle_data, PuzzleResponse
from config import MODEL_CONFIGS

RESPONSE_TEMPLATE = "JSON Output:\n"

def generate_prompt(puzzle: ChessPuzzle) -> str:
    return f"""Analyze the following chess position and output the best sequence of moves: {puzzle.fen}
{RESPONSE_TEMPLATE}"""

def main():
    parser = argparse.ArgumentParser(
        description='Generate responses for chess puzzles using a language model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python generate.py --puzzle_file data/wtharvey-sample.json --model-config llama --num-problems 10'
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
    args = parser.parse_args()
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.backends.mps.is_available():
        print(f"Emptying MPS cache")
        torch.mps.empty_cache()
    
    # Get model configuration
    if args.model_config not in MODEL_CONFIGS:
        print(f"Error: Unknown model config '{args.model_config}'. Available: {list(MODEL_CONFIGS.keys())}", file=sys.stderr)
        return 1
    model_config = MODEL_CONFIGS[args.model_config]
    base_model_name = model_config["name"]
    
    # Use bfloat16 on CUDA (matches training), float16 on MPS/CPU
    model_dtype = torch.bfloat16 if device == "cuda" else torch.float16

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
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.trained_model_dir:
        print(f"Loading LoRA adapters: {args.trained_model_dir}")
        model = PeftModel.from_pretrained(
            model,
            args.trained_model_dir,
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

        prompt_batch = [generate_prompt(puzzle) for puzzle in batch_puzzles]
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
    all_prompts = [generate_prompt(puzzle) for puzzle in puzzles_to_process]

    for puzzle, prompt, output in zip(puzzles_to_process, all_prompts, all_generated_texts):
        output_truncated = output.replace(prompt, '').strip()
        
        print(f"\nPuzzle: {puzzle.description}")
        print(f"FEN: {puzzle.fen}")
        print(f"Expected solution: {puzzle.solution}")
        
        try:
            model_response = PuzzleResponse.model_validate(json.loads(output_truncated))
            print(f"Model solution: {model_response.solution}")
            correct = model_response.solution == puzzle.solution
            print(f"Correct: {correct}")
            
            result = {
                "fen": puzzle.fen,
                "description": puzzle.description,
                "citation": puzzle.citation,
                "expected_solution": puzzle.solution,
                "model_solution": model_response.solution,
                "correct": correct
            }
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"Error: Invalid JSON response: {e}")
            print(f"Output: {output_truncated}")
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
    num_correct = sum(1 for r in results if r.get('correct', False))
    num_incorrect = sum(1 for r in results if 'correct' in r and not r['correct'])
    num_errors = sum(1 for r in results if 'error' in r)
    print(f"Correct solutions: {num_correct}")
    print(f"Incorrect solutions: {num_incorrect}")
    print(f"Errors: {num_errors}")
    if len(results) > 0:
        print(f"Accuracy: {num_correct / len(results):.2%}")
    print(f"Total generation time: {elapsed_time:.2f} seconds")
    print(f"Average time per puzzle: {elapsed_time / num_problems:.2f} seconds")
    
    # Save results if output file specified
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

