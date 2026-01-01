#!/usr/bin/env python3
"""
Send chess puzzles to OpenRouter API for analysis.
"""

import argparse
import json
import sys
from pathlib import Path
import requests
import re
import time


def strip_markdown_json(text):
    """
    Strip markdown code blocks from JSON response.
    Handles cases like ```json ... ``` or ``` ... ```
    """
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace
    return text.strip()


def get_openrouter_api_key():
    """
    Read OpenRouter API key from .openrouter_api_key file.
    
    Returns:
        str: The API key
        
    Raises:
        FileNotFoundError: If .openrouter_api_key file doesn't exist
        ValueError: If the API key file is empty
    """
    api_key_file = Path('.openrouter_api_key')
    if not api_key_file.exists():
        raise FileNotFoundError(
            ".openrouter_api_key file not found. "
            "Please create a .openrouter_api_key file with your API key."
        )
    
    api_key = api_key_file.read_text().strip()
    if not api_key:
        raise ValueError("API key file (.openrouter_api_key) is empty")
    
    return api_key


def create_puzzle_prompt(fen):
    """
    Create a prompt for analyzing a chess puzzle.
    
    Args:
        fen (str): FEN notation of the position
        
    Returns:
        str: The formatted prompt
    """
    prompt = f"""You are analyzing a chess puzzle position.

POSITION (FEN): {fen}

Your task is to find the best sequence of moves for this position. Analyze the position carefully and determine:
1. The best sequence of moves (in standard algebraic notation, e.g., "1. e4 e5 2. Nf3")
2. Why these moves are the best (tactical themes, strategic considerations, etc.)
3. Your confidence level in this solution

Respond with ONLY valid JSON (no markdown formatting):
{{
  "best_moves": "sequence of moves in SAN notation (e.g., '1. e4 e5 2. Nf3 Nc6')",
  "description": "detailed explanation of why these moves are best, including tactical themes and strategic considerations",
  "confidence": "high/medium/low"
}}
"""
    return prompt


def send_to_openrouter(model, prompt, api_key):
    """
    Send a prompt to OpenRouter API.
    
    Args:
        model (str): Model name (e.g., "meta-llama/Llama-3.1-8B-Instruct", "qwen/qwen3-8b"
        prompt (str): The prompt to send
        api_key (str): OpenRouter API key
        
    Returns:
        str: The response text from the API
        
    Raises:
        Exception: If the API call fails
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/chess_advisor",  # Optional
        "X-Title": "Chess Advisor"  # Optional
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(
        description='Send chess puzzles to OpenRouter API for analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python eval_puzzles.py data/wtharvey-mate-in-2.json meta-llama/Llama-3.1-8B-Instruct --num-problems 5'
    )
    parser.add_argument(
        'puzzle_file',
        type=str,
        help='Path to the JSON puzzle file (e.g., data/wtharvey-mate-in-2.json)'
    )
    parser.add_argument(
        'model',
        type=str,
        help='Model name (e.g., meta-llama/Llama-3.1-8B-Instruct, qwen/qwen3-8b'
    )
    parser.add_argument(
        '--num-problems',
        type=int,
        default=1,
        help='Number of problems to send (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save results (default: prints to stdout)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API calls in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Read API key
    try:
        api_key = get_openrouter_api_key()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Read puzzle file
    puzzle_file = Path(args.puzzle_file)
    if not puzzle_file.exists():
        print(f"Error: Puzzle file not found: {puzzle_file}", file=sys.stderr)
        return 1
    
    try:
        with open(puzzle_file, 'r') as f:
            puzzles = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in puzzle file: {e}", file=sys.stderr)
        return 1
    
    # Limit number of problems
    num_problems = min(args.num_problems, len(puzzles))
    puzzles_to_process = puzzles[:num_problems]
    
    print(f"Processing {num_problems} puzzles using model: {args.model}")
    print(f"Source file: {puzzle_file.name}")
    
    results = []
    
    for i, puzzle in enumerate(puzzles_to_process, 1):
        fen = puzzle['fen']
        description = puzzle['description']
        solution = puzzle['solution']
        citation = puzzle.get('citation', 'Unknown')
        
        print(f"\n[{i}/{num_problems}] Processing puzzle: {citation}")
        print(f"FEN: {fen}")
        print(f"Expected solution: {solution}")
        
        try:
            prompt = create_puzzle_prompt(fen)
            response_text = send_to_openrouter(args.model, prompt, api_key)
            
            # Clean and parse JSON response
            cleaned_response = strip_markdown_json(response_text)
            try:
                response_json = json.loads(cleaned_response)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON response. Raw response:")
                print(response_text)
                response_json = {"raw_response": response_text}
            
            result = {
                "puzzle_index": i,
                "citation": citation,
                "fen": fen,
                "description": description,
                "expected_solution": puzzle.get('solution', 'N/A'),
                "model_solution": response_json.get('best_moves', 'N/A'),
                "model_confidence": response_json.get('confidence', 'N/A'),
                "model_description": response_json.get('description', 'N/A')
            }

            print(f"Model solution:    {result['model_solution']}")
            print(f"Description:       {result['model_description']}...")
            if result['model_solution'] == result['expected_solution']:
                result['correct'] = True
            else:
                result['correct'] = False
            
            results.append(result)

            # Delay between requests to avoid rate limiting
            if i < num_problems:
                time.sleep(args.delay)
                
        except Exception as e:
            print(f"Error processing puzzle {i}: {e}", file=sys.stderr)
            result = {
                "puzzle_index": i,
                "citation": citation,
                "fen": fen,
                "error": str(e)
            }
            results.append(result)
    
    # Output results
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent='\t', ensure_ascii=False)
        print(f"\n\nResults saved to: {output_file}")
    else:
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        num_correct = sum(1 for result in results if 'correct' in result and result['correct'])
        num_incorrect = sum(1 for result in results if 'correct' in result and not result['correct'])
        num_errors = sum(1 for result in results if 'error' in result)
        print(f"Correct solutions: {num_correct}")
        print(f"Incorrect solutions: {num_incorrect}")
        print(f"Errors: {num_errors}")
        print(f"Accuracy: {num_correct / len(results)}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

