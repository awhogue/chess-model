#!/usr/bin/env python3
"""
Utility functions for chess puzzle processing.
"""

import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel


class PuzzleResponse(BaseModel):
    """Response model for puzzle analysis."""
    best_moves: str
    description: str
    confidence: str

class ChessPuzzle(BaseModel):
    """Single chess puzzle."""
    fen: str
    solution: str
    description: str
    citation: str
    source: str

def load_puzzle_data(puzzle_file: str) -> List[Dict]:
    """
    Load chess puzzle data from JSON file.
    
    Args:
        puzzle_file: Path to JSON file containing puzzles in ChessPuzzle format.
        num_puzzles: Number of puzzles to load. If -1, load all puzzles.
    Returns:
        List of ChessPuzzle objects
        
    Raises:
        FileNotFoundError: If puzzle file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    if not Path(puzzle_file).exists():
        raise FileNotFoundError(f"Puzzle file not found: {puzzle_file}")
    
    with Path(puzzle_file).open('r') as f:
        puzzles = [ChessPuzzle.model_validate(puzzle) for puzzle in json.load(f)]

    return puzzles


def full_puzzle_prompt(fen: str) -> str:
    """
    Create a prompt for analyzing a chess puzzle.
    
    Args:
        fen: FEN notation of the position

    Returns:
        The formatted prompt string
    """
    prompt = f"""You are an expert chess player analyzing a chess position.

POSITION (FEN): {fen}

Your task is to find the best sequence of moves for this position. Analyze the position carefully and determine:
1. The best sequence of up to 10 moves (in standard algebraic notation, e.g., "1. e4 e5 2. Nf3")
2. Why these moves are the best (tactical themes, strategic considerations, etc.)
3. Your confidence level in this solution

You must respond with ONLY valid JSON in the following format (absolutely no markdown formatting or other types of text):
{PuzzleResponse.model_json_schema()}
"""
    return prompt

