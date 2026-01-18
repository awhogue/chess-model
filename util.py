#!/usr/bin/env python3
"""
Utility functions for chess puzzle processing.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel


class PuzzleResponse(BaseModel):
    """Response model for puzzle analysis."""
    solution: str
    # description: str
    # confidence: str

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


@dataclass
class SolutionComparison:
    """Result of comparing expected vs model solution."""
    exact_match: bool
    normalized_match: bool  # Match after stripping move numbers
    first_move_correct: bool
    correct_moves: int
    total_moves: int
    partial_score: float  # correct_moves / total_moves
    expected_moves: List[str]
    model_moves: List[str]


def normalize_solution(solution: str) -> List[str]:
    """
    Normalize a chess solution string into a list of individual moves.

    Removes move numbers (e.g., "1.", "2.") and splits into individual moves.
    Handles formats like:
      - "1. Qxf7+ Rxf7 2. Re8#"
      - "Qxf7+ Rxf7 Re8#"
      - "1.e4 e5 2.Nf3"

    Returns:
        List of moves in order, e.g., ["Qxf7+", "Rxf7", "Re8#"]
    """
    # Remove move numbers like "1.", "2.", "1..." (for black's move)
    cleaned = re.sub(r'\d+\.+\s*', ' ', solution)
    # Split on whitespace and filter empty strings
    moves = [m.strip() for m in cleaned.split() if m.strip()]
    return moves


def compare_solutions(expected: str, model: str) -> SolutionComparison:
    """
    Compare expected solution with model's solution.

    Handles:
    - Move number normalization ("1. e4 e5" vs "e4 e5")
    - Partial credit for getting some moves correct
    - First move correctness

    Args:
        expected: Expected solution string (e.g., "1. Qxf7+ Rxf7 2. Re8#")
        model: Model's solution string (e.g., "Qxf7+ Rxf7 Re8#")

    Returns:
        SolutionComparison with detailed comparison results
    """
    expected_moves = normalize_solution(expected)
    model_moves = normalize_solution(model)

    # Exact string match (after basic whitespace normalization)
    exact_match = expected.strip() == model.strip()

    # Normalized match (same moves after stripping move numbers)
    normalized_match = expected_moves == model_moves

    # Count correct moves (in sequence from the start)
    correct_moves = 0
    for exp, mod in zip(expected_moves, model_moves):
        if exp == mod:
            correct_moves += 1
        else:
            break  # Stop at first mismatch

    # First move correct
    first_move_correct = (
        len(expected_moves) > 0 and
        len(model_moves) > 0 and
        expected_moves[0] == model_moves[0]
    )

    # Partial score based on expected moves
    total_moves = len(expected_moves)
    partial_score = correct_moves / total_moves if total_moves > 0 else 0.0

    return SolutionComparison(
        exact_match=exact_match,
        normalized_match=normalized_match,
        first_move_correct=first_move_correct,
        correct_moves=correct_moves,
        total_moves=total_moves,
        partial_score=partial_score,
        expected_moves=expected_moves,
        model_moves=model_moves,
    )

