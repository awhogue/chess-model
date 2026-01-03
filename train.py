#!/usr/bin/env python3

import argparse
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from util import load_puzzle_data, create_puzzle_prompt, PuzzleResponse


def get_huggingface_api_key():
    """
    Read Hugging Face API key from .huggingface_api_key file.
    
    Returns:
        str: The API key
        
    Raises:
        FileNotFoundError: If .huggingface_api_key file doesn't exist
        ValueError: If the API key file is empty
    """
    api_key_file = Path('.huggingface_api_key')
    if not api_key_file.exists():
        raise FileNotFoundError(
            ".huggingface_api_key file not found. "
            "Please create a .huggingface_api_key file with your API key."
        )
    
    api_key = api_key_file.read_text().strip()
    if not api_key:
        raise ValueError("API key file (.huggingface_api_key) is empty")
    
    return api_key

def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model on chess puzzles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python train.py data/wtharvey-mate-in-2.json --model-name meta-llama/Llama-3.2-3B'
    )
    parser.add_argument(
        'puzzle_file',
        type=str,
        help='Path to JSON puzzle file (e.g., data/wtharvey-mate-in-2.json)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Llama-3.2-3B',
        help='Hugging Face model name or path (default: meta-llama/Llama-3.2-3B)'
    )
    parser.add_argument(
        '--num-problems',
        type=int,
        default=1,
        help='Number of problems to train on (default: 1)'
    )
    args = parser.parse_args()
    
    puzzles = load_puzzle_data(args.puzzle_file)
    print(f"Loaded {len(puzzles)} puzzles from {args.puzzle_file}")
    
    # Limit number of problems
    if args.num_problems == -1:
        num_problems = len(puzzles)
    else:
        num_problems = min(args.num_problems, len(puzzles))
    puzzles_to_process = puzzles[:num_problems]
    
    login(get_huggingface_api_key())

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 

    prompt_batch = [create_puzzle_prompt(puzzle['fen']) for puzzle in puzzles_to_process]
    inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start_time = time.perf_counter()

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_text_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    for puzzle, prompt, output in zip(puzzles_to_process, prompt_batch, generated_text_batch):
        model_response = PuzzleResponse.model_validate(json.loads(output[len(prompt):]))
        print(f"Puzzle: {puzzle['description']}")
        print(f"Solution: {puzzle['solution']}")
        print(f"Model:    {model_response.best_moves}")
        
    elapsed_time = time.perf_counter() - start_time
    print(f"Total generation time: {elapsed_time:.2f} seconds ({elapsed_time / num_problems:.2f} seconds per puzzle)")

    return 0

if __name__ == "__main__":
    main()