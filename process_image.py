#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import io
from PIL import Image
import google.genai as genai
from google.genai import types
import chess
import json
import re

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

def get_api_key(api_key_arg=None):
    """
    Get API key from argument or .gemini_api_key file.
    
    Args:
        api_key_arg (str, optional): API key provided as command-line argument
        
    Returns:
        str: The API key
        
    Raises:
        FileNotFoundError: If api_key_arg is None and .gemini_api_key file doesn't exist
        ValueError: If the API key file is empty
    """
    if api_key_arg:
        return api_key_arg
    
    api_key_file = Path('.gemini_api_key')
    if not api_key_file.exists():
        raise FileNotFoundError(
            "API key not provided and .gemini_api_key file not found. "
            "Either provide the API key as an argument or create a .gemini_api_key file."
        )
    
    api_key = api_key_file.read_text().strip()
    if not api_key:
        raise ValueError("API key file (.gemini_api_key) is empty")
    
    return api_key

def send_image_to_gemini(image_path, prompt, api_key):
    """
    Send an image to the Gemini 2.0 Flash API with a specified prompt.
    
    Args:
        image (PIL.Image.Image): The PIL Image object to send
        prompt (str): The text prompt to send with the image
        api_key (str): The Google Gemini API key
        
    Returns:
        str: The response text from the Gemini API
        
    Raises:
        Exception: If the API call fails
    """
    # Create a client with the API key
    client = genai.Client(api_key=api_key)
    
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type='image/png')
        ],
    )
    
    return response.text

FEN_PROMPT = """You are analyzing a chess board image to extract the exact position in FEN notation.

CRITICAL INSTRUCTIONS:
1. Carefully examine EACH SQUARE on the board, going rank by rank from rank 8 to rank 1
2. For each piece, identify if it's white (uppercase) or black (lowercase)
3. Piece symbols: K/k=king, Q/q=queen, R/r=rook, B/b=bishop, N/n=knight, P/p=pawn
4. In FEN: each rank separated by /, empty squares shown as numbers (1-8)
5. Determine whose turn it is based on the position or context
6. Include full FEN: position activeColor castlingRights enPassant halfmove fullmove

BOARD ORIENTATION:
- Determine if board shows white's perspective (rank 1 at bottom) or black's (rank 8 at bottom)
- If it's black's perspective, you MUST flip the position when writing FEN

ACCURACY TIPS:
- Take time to distinguish bishops from knights
- Count empty squares carefully
- Verify your FEN has exactly 8 ranks
- Double-check piece colors

Example starting position FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1

Respond with ONLY valid JSON (no markdown formatting):
{
  "fen": "complete FEN string",
  "turn": "white" or "black",
  "position_description": "brief position description",
  "board_orientation": "white" or "black",
  "confidence": "high/medium/low",
  "best_move": "best move in SAN notation"
  "best_move_confidence": "high/medium/low",
  "best_move_description": "brief description of the best move",
}
"""

def main():
    parser = argparse.ArgumentParser(
        description='Process a PNG image file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python process_image.py image.png [api_key]'
    )
    parser.add_argument(
        'image_file',
        type=str,
        help='Path to the PNG image file to process'
    )
    parser.add_argument(
        'api_key',
        type=str,
        nargs='?',
        default=None,
        help='API key for the Gemini API (optional, will read from .gemini_api_key if not provided)'
    )
    args = parser.parse_args()
    
    try:
        api_key = get_api_key(args.api_key)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    response = send_image_to_gemini(args.image_file, FEN_PROMPT, api_key)
    # Strip markdown code blocks if present
    cleaned_response = strip_markdown_json(response)
    json_response = json.loads(cleaned_response)
    fen = json_response['fen']
    print(fen)
    board = chess.Board(fen)
    print(board)
    print(json_response['best_move'])
    print(json_response['best_move_description'])
   
    return 0
    

if __name__ == '__main__':
    sys.exit(main())

