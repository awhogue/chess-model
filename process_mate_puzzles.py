#!/usr/bin/env python3
"""
Process mate puzzle files into JSON format.
"""

import json
import re
import sys
from pathlib import Path


def parse_mate_file(file_path, mate_depth=None):
    """
    Parse a mate puzzle file and extract entries.
    
    Args:
        file_path (Path): Path to the input file
        mate_depth (int, optional): Mate depth (2, 3, 4, etc.). If None, tries to extract from filename.
        
    Returns:
        tuple: (source_url, list of puzzle dictionaries)
    """
    content = file_path.read_text()
    lines = content.split('\n')
    
    # Extract source URL from first line
    source_url = lines[0].replace('Source: ', '').strip()
    
    # Try to extract mate depth from filename if not provided
    if mate_depth is None:
        match = re.search(r'mate-in-(\d+)', file_path.name)
        if match:
            mate_depth = int(match.group(1))
        else:
            # Default to 2 if can't determine
            mate_depth = 2
    
    puzzles = []
    i = 1  # Start after the source line
    
    while i < len(lines):
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        if i >= len(lines):
            break
        
        # Citation line (player names, location, year)
        citation = lines[i].strip()
        i += 1
        
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        if i >= len(lines):
            break
        
        # FEN line
        fen = lines[i].strip()
        i += 1
        
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        if i >= len(lines):
            break
        
        # Solution line
        solution = lines[i].strip()
        i += 1
        
        # Determine if it's white or black to move based on solution
        if solution.startswith('1...'):
            description = f"Black Mates in {mate_depth}."
        else:
            description = f"White Mates in {mate_depth}."
        
        puzzle = {
            "fen": fen,
            "solution": solution,
            "description": description,
            "citation": citation,
            "source": source_url
        }
        
        puzzles.append(puzzle)
    
    return source_url, puzzles


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_mate_puzzles.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Default output filename
    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = input_file.parent / f"{input_file.stem}.json"
    
    try:
        source_url, puzzles = parse_mate_file(input_file)
        print(f"Parsed {len(puzzles)} puzzles from {input_file.name}")
        print(f"Source: {source_url}")
        
        # Write JSON output
        with open(output_file, 'w') as f:
            json.dump(puzzles, f, indent='\t', ensure_ascii=False)
        
        print(f"Output written to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

