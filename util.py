#!/usr/bin/env python3
"""
Utility functions for chess puzzle processing.
"""

import json
import re
import shutil
import chess
import chess.engine
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
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


def find_stockfish(stockfish_path: Optional[str] = None) -> str:
    """
    Find the Stockfish binary path.

    Searches shutil.which() first, then common installation paths.

    Args:
        stockfish_path: Explicit path to Stockfish binary (returned directly if provided)

    Returns:
        Path to Stockfish binary

    Raises:
        FileNotFoundError: If Stockfish is not found
    """
    if stockfish_path:
        return stockfish_path

    found = shutil.which("stockfish")
    if found:
        return found

    common_paths = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/snap/bin/stockfish",
    ]
    for path in common_paths:
        if Path(path).is_file():
            return path

    raise FileNotFoundError(
        "Stockfish not found. Install it:\n"
        "  macOS:         brew install stockfish\n"
        "  Ubuntu/Debian: apt install stockfish\n"
        "  Other:         https://stockfishchess.org/download/\n"
        "Or pass --stockfish-path /path/to/stockfish"
    )


class StockfishManager:
    """Context manager for a reusable Stockfish engine instance."""

    def __init__(self, stockfish_path: str, depth: int = 15, time_limit: float = 0.1):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def start(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": 1, "Hash": 64})

    def stop(self):
        if self.engine:
            try:
                self.engine.quit()
            except chess.engine.EngineTerminatedError:
                pass
            self.engine = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def _ensure_engine(self):
        """Restart engine if it has terminated."""
        if self.engine is None:
            self.start()

    def evaluate_position(self, board: chess.Board) -> int:
        """
        Evaluate a position from the current side to move's perspective.

        Returns centipawn score (mate scores clamped to +/-10000).
        """
        self._ensure_engine()
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth, time=self.time_limit),
        )
        score = info["score"].relative
        if score.is_mate():
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000
        return score.score()

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Return Stockfish's best move for the position."""
        self._ensure_engine()
        result = self.engine.play(
            board,
            chess.engine.Limit(depth=self.depth, time=self.time_limit),
        )
        return result.move

