#!/usr/bin/env python3

import torch
import argparse
import time
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login
from datasets import Dataset

from util import full_puzzle_prompt, ChessPuzzle, load_puzzle_data


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

def generate(model, tokenizer, puzzles):
    """
    Generate responses for a batch of puzzles.
    
    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer for the model
        puzzles: List of ChessPuzzle objects
        
    Returns:
        tuple: (generated_text_batch, elapsed_time) where generated_text_batch is a list of generated text strings
    """
    prompt_batch = [f"{full_puzzle_prompt(puzzle.fen)}\nJSON Output:\n" for puzzle in puzzles]
    inputs = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start_time = time.perf_counter()

    generated_ids = model.generate(**inputs, 
        max_new_tokens=500, repetition_penalty=1.5,  # Higher = more penalty for repeating
        no_repeat_ngram_size=3,  # Don't repeat 3-grams
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id)
    generated_text_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    elapsed_time = time.perf_counter() - start_time
    
    return generated_text_batch, elapsed_time

def training_example(puzzle: ChessPuzzle, tokenizer: AutoTokenizer) -> str:
    """
    Create a training example from a puzzle.
    """
    prompt = f"""Analyze the following chess position and output the best sequence of moves: {puzzle.fen}
JSON Output:
{{ "solution": "{puzzle.solution}" }}{tokenizer.eos_token}"""
    return {"text": prompt}


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a language model on chess puzzles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python train.py --puzzle_file data/wtharvey-sample.json --model-name meta-llama/Llama-3.2-3B'
    )
    parser.add_argument(
        '--puzzle_file',
        type=str,
        default='data/wtharvey-sample.json',
        help='Path to JSON puzzle file (e.g., data/wtharvey-sample.json)'
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
        default=100,
        help='Number of problems to use (default: 100). This will be split into a training and test set.'
    )
    parser.add_argument(
        '--output-model-dir',
        type=str,
        default=f"models/trained/",
        help='Output directory for the fine-tuned model (default: models/<model_name>)'
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    login(get_huggingface_api_key())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map=device)
    tokenizer.pad_token = tokenizer.eos_token

    puzzles = load_puzzle_data(args.puzzle_file)[:args.num_problems]
    puzzle_examples = [ training_example(puzzle, tokenizer) for puzzle in puzzles ]
    puzzles_dataset = Dataset.from_list(puzzle_examples).train_test_split(test_size=0.1)
    print(puzzles_dataset)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16, device_map=device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=args.output_model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=puzzles_dataset["train"],
        eval_dataset=puzzles_dataset["test"],
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_model_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())