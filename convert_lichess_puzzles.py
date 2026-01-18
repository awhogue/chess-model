#!/usr/bin/env python3
"""
Convert Lichess puzzle CSV to wtharvey JSON format.

Thanks Claude code!

Usage:
    python convert_lichess_puzzles.py data/lichess_db_puzzle.csv -o output.json -n 1000
    python convert_lichess_puzzles.py data/lichess_db_puzzle.csv -o output.json -n 500 --themes mateIn2
"""

import argparse
import csv
import json
import sys
from typing import Optional

import chess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Lichess puzzle CSV to wtharvey JSON format"
    )
    parser.add_argument("input_csv", help="Path to Lichess puzzle CSV file")
    parser.add_argument(
        "-o", "--output", default="puzzles.json", help="Output JSON file path"
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None, help="Maximum number of puzzles to extract"
    )
    parser.add_argument(
        "--themes",
        type=str,
        default=None,
        help="Filter by theme(s). Comma-separated for multiple themes (puzzle must match ALL)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N puzzles before outputting (default: 0)",
    )
    return parser.parse_args()


def uci_to_san(board: chess.Board, uci_move: str) -> str:
    """Convert a UCI move to standard algebraic notation."""
    move = chess.Move.from_uci(uci_move)
    return board.san(move)


def apply_move(board: chess.Board, uci_move: str) -> None:
    """Apply a UCI move to the board."""
    move = chess.Move.from_uci(uci_move)
    board.push(move)


def convert_puzzle(row: dict) -> Optional[dict]:
    """
    Convert a single Lichess puzzle to wtharvey format.

    The Lichess puzzle format has the position BEFORE the opponent's move,
    so we apply the first move to get the actual puzzle position.
    The remaining moves are the solution.
    """
    try:
        fen = row["FEN"]
        moves_uci = row["Moves"].split()
        themes = row["Themes"]
        game_url = row["GameUrl"]
        rating = row["Rating"]
        puzzle_id = row["PuzzleId"]

        if len(moves_uci) < 2:
            return None

        # Create board from initial FEN
        board = chess.Board(fen)

        # Apply the first move (opponent's move) to get the puzzle starting position
        opponent_move = moves_uci[0]
        apply_move(board, opponent_move)

        # The new FEN after opponent's move is the puzzle position
        puzzle_fen = board.fen()

        # Convert remaining moves (the solution) from UCI to SAN
        solution_moves = []
        for uci_move in moves_uci[1:]:
            san_move = uci_to_san(board, uci_move)
            solution_moves.append(san_move)
            apply_move(board, uci_move)

        # Format solution as space-separated string
        solution = " ".join(solution_moves)

        # Determine description based on themes
        if "mate" in themes.lower():
            description = "Checkmate"
        else:
            description = "Winning moves"

        return {
            "fen": puzzle_fen,
            "solution": solution,
            "description": description,
            "citation": f"Lichess puzzle {puzzle_id} (Rating: {rating})",
            "source": game_url,
            "themes": themes,
        }
    except Exception as e:
        print(f"Error converting puzzle {row.get('PuzzleId', 'unknown')}: {e}", file=sys.stderr)
        return None


def matches_themes(puzzle_themes: str, filter_themes: list[str]) -> bool:
    """Check if puzzle contains ALL specified themes."""
    puzzle_theme_set = set(puzzle_themes.split())
    return all(theme in puzzle_theme_set for theme in filter_themes)


def main():
    args = parse_args()

    # Parse theme filter if provided
    filter_themes = None
    if args.themes:
        filter_themes = [t.strip() for t in args.themes.split(",")]

    print(f"Reading puzzles from {args.input_csv}...")

    # Read all puzzles (or filtered by theme)
    puzzles = []
    with open(args.input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Apply theme filter if specified
            if filter_themes and not matches_themes(row["Themes"], filter_themes):
                continue
            puzzles.append(row)

    print(f"Found {len(puzzles)} puzzles" + (f" matching themes: {filter_themes}" if filter_themes else ""))

    # Sort by popularity (descending) and take top N
    puzzles.sort(key=lambda x: int(x["Popularity"]), reverse=True)

    if args.limit:
        puzzles = puzzles[:args.limit]
        print(f"Taking top {len(puzzles)} most popular puzzles")

    # Skip first N puzzles if specified
    if args.skip > 0:
        if args.skip >= len(puzzles):
            print(f"Warning: --skip {args.skip} is >= total puzzles {len(puzzles)}, no puzzles will be output", file=sys.stderr)
            puzzles = []
        else:
            puzzles = puzzles[args.skip:]
            print(f"Skipped first {args.skip} puzzles, {len(puzzles)} remaining")

    # Convert puzzles
    converted = []
    for row in puzzles:
        result = convert_puzzle(row)
        if result:
            converted.append(result)

    print(f"Successfully converted {len(converted)} puzzles")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
