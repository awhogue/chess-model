#!/usr/bin/env python3
"""
Scrape chess puzzles from wtharvey.com website.
Supports both HTML pages and text files.

Usage:
    python scrape_wtharvey.py <url1> [url2 ...] [-o output.json]

Examples:
    # Scrape a single HTML page
    python scrape_wtharvey.py https://wtharvey.com/arkh.html

    # Scrape a text file
    python scrape_wtharvey.py https://wtharvey.com/m8n4.txt

    # Scrape multiple pages (can mix HTML and text files)
    python scrape_wtharvey.py arkh.html 1994.html -o puzzles.json

    # Scrape all mate-in-N puzzle files
    python scrape_wtharvey.py m8n2.txt m8n3.txt m8n4.txt -o mate_puzzles.json

Output format:
    Each puzzle is a JSON object with the following fields:
    - fen: FEN notation of the position
    - solution: The move sequence
    - description: Description like "White Mates in 4."
    - citation: Game citation (players, location, year)
    - source: URL of the source page
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class WTHarveyPuzzleScraper:
    """Scraper for chess puzzles from wtharvey.com"""

    BASE_URL = "https://wtharvey.com/"

    def __init__(self, delay=1.0):
        """
        Initialize the scraper.

        Args:
            delay (float): Delay in seconds between requests to be polite
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; ChessPuzzleScraper/1.0)'
        })

    def scrape_text_file(self, url):
        """
        Scrape puzzles from a text file format.

        Args:
            url (str): URL of the text file

        Returns:
            list: List of puzzle dictionaries
        """
        print(f"Fetching text file: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        lines = response.text.split('\n')
        puzzles = []
        i = 0

        # Extract mate depth from URL if possible
        mate_match = re.search(r'm8n(\d+)', url)
        mate_depth = int(mate_match.group(1)) if mate_match else None

        # Pattern to match FEN notation
        fen_pattern = re.compile(r'^[rnbqkpRNBQKP1-8/\s]+ [wb] [-KQkq]+ [-a-h0-8]+ \d+ \d+$')

        # Pattern to match citation (player vs player, location, year)
        citation_pattern = re.compile(r'.+\s+vs\s+.+,\s*.+,\s*\d{4}', re.IGNORECASE)

        # Pattern to match move notation (solution)
        solution_pattern = re.compile(r'^(\d+\.+|\d+\.\.\.).*[#\s]')

        while i < len(lines):
            # Skip blank lines
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                break

            line = lines[i].strip()

            # Skip source line if present
            if line.startswith('Source:'):
                i += 1
                continue

            # Check if this looks like a citation
            if not citation_pattern.match(line):
                # Skip lines that don't look like citations (header text, etc.)
                i += 1
                continue

            # Citation line (player names, location, year)
            citation = line
            i += 1

            # Skip blank lines
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                break

            # FEN line
            fen = lines[i].strip()

            # Validate it looks like a FEN
            if not fen_pattern.match(fen):
                # If it doesn't look like a FEN, this wasn't a real puzzle entry
                continue

            i += 1

            # Skip blank lines
            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                break

            # Solution line
            solution = lines[i].strip()

            # Validate it looks like a solution
            if not solution_pattern.match(solution):
                # If it doesn't look like a solution, skip
                continue

            i += 1

            # Determine if it's white or black to move based on solution
            if mate_depth:
                if solution.startswith('1...'):
                    description = f"Black Mates in {mate_depth}."
                else:
                    description = f"White Mates in {mate_depth}."
            else:
                # Try to infer from solution
                description = "Mates in ?"

            puzzle = {
                "fen": fen,
                "solution": solution,
                "description": description,
                "citation": citation,
                "source": url
            }

            puzzles.append(puzzle)

        return puzzles

    def scrape_html_page(self, url):
        """
        Scrape puzzles from an HTML page.

        Args:
            url (str): URL of the HTML page

        Returns:
            list: List of puzzle dictionaries
        """
        print(f"Fetching HTML page: {url}")
        response = self.session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        puzzles = []
        seen_fens = set()  # Track FENs we've already added to avoid duplicates

        # Get all text and combine into one string, then split by common delimiters
        # Use separator to ensure line breaks become spaces
        text = soup.get_text(separator='\n')

        # Pattern to match FEN notation
        # FEN format: 8 ranks separated by /, followed by side to move, castling, en passant, halfmove, fullmove
        fen_pattern = re.compile(r'([rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7})\s+([wb])\s+([-KQkq]+)\s+([-a-h0-8]+)\s+(\d+)\s+(\d+)')

        # Pattern to match solutions in brackets (allow newlines inside brackets)
        solution_pattern = re.compile(r'\[\s*(.+?)\s*\]', re.DOTALL)

        # Pattern to match mate descriptions
        # Format: "White mates in 2" or "Black mates in 3"
        mate_desc_pattern = re.compile(r'(White|Black)\s+mates?\s+in\s+(\d+)', re.IGNORECASE)

        # Pattern to match citation without explicit mate description
        # Format: "Player vs Player, Location, Year"
        citation_pattern = re.compile(r'\.?\s*([A-Za-z\s\.]+?)\s+vs\s+([A-Za-z\s\.]+?),\s*([^,]+),\s*(\d{4})')

        # Try to find patterns in the text
        # Split text into manageable chunks (by line or by some pattern)
        lines = text.split('\n')

        # Process line by line, looking for patterns
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for mate description
            mate_match = mate_desc_pattern.search(line)
            if mate_match:
                color = mate_match.group(1)
                moves = mate_match.group(2)
                description = f"{color} Mates in {moves}."

                # Look for citation in nearby lines
                citation = ""
                for j in range(max(0, i-2), min(i+5, len(lines))):
                    citation_match = citation_pattern.search(lines[j])
                    if citation_match:
                        player1 = citation_match.group(1).strip()
                        player2 = citation_match.group(2).strip()
                        location = citation_match.group(3).strip()
                        year = citation_match.group(4)
                        citation = f"{player1} vs {player2}, {location}, {year}"
                        break

                # Look for FEN in nearby lines
                fen = None
                for j in range(max(0, i-2), min(i+5, len(lines))):
                    fen_match = fen_pattern.search(lines[j])
                    if fen_match:
                        fen = fen_match.group(0)
                        break

                # Look for solution in nearby lines (join lines to handle multi-line brackets)
                solution = None
                nearby_text = '\n'.join(lines[max(0, i-2):min(i+5, len(lines))])
                sol_match = solution_pattern.search(nearby_text)
                if sol_match:
                    # Remove newlines and extra whitespace from solution
                    solution = ' '.join(sol_match.group(1).split())

                if fen and solution and fen not in seen_fens:
                    seen_fens.add(fen)
                    puzzle = {
                        "fen": fen,
                        "solution": solution,
                        "description": description,
                        "citation": citation,
                        "source": url
                    }
                    puzzles.append(puzzle)
            else:
                # Try to match just citation and look for nearby FEN and solution
                citation_match = citation_pattern.search(line)
                if citation_match:
                    player1 = citation_match.group(1).strip()
                    player2 = citation_match.group(2).strip()
                    location = citation_match.group(3).strip()
                    year = citation_match.group(4)

                    citation = f"{player1} vs {player2}, {location}, {year}"

                    # Look for FEN in nearby lines
                    fen = None
                    for j in range(max(0, i-2), min(i+5, len(lines))):
                        fen_match = fen_pattern.search(lines[j])
                        if fen_match:
                            fen = fen_match.group(0)
                            break

                    # Look for solution in nearby lines (join lines to handle multi-line brackets)
                    solution = None
                    nearby_text = '\n'.join(lines[max(0, i-2):min(i+5, len(lines))])
                    sol_match = solution_pattern.search(nearby_text)
                    if sol_match:
                        # Remove newlines and extra whitespace from solution
                        solution = ' '.join(sol_match.group(1).split())

                    if fen and solution and fen not in seen_fens:
                        seen_fens.add(fen)
                        # Try to determine description from context or solution
                        description = "Winning moves"

                        puzzle = {
                            "fen": fen,
                            "solution": solution,
                            "description": description,
                            "citation": citation,
                            "source": url
                        }
                        puzzles.append(puzzle)

        return puzzles

    def scrape_url(self, url):
        """
        Scrape puzzles from a URL (auto-detects format).

        Args:
            url (str): URL to scrape

        Returns:
            list: List of puzzle dictionaries
        """
        # Normalize URL
        if not url.startswith('http'):
            url = urljoin(self.BASE_URL, url)

        try:
            # Determine if it's a text file or HTML based on extension
            if url.endswith('.txt'):
                puzzles = self.scrape_text_file(url)
            else:
                puzzles = self.scrape_html_page(url)

            print(f"  → Found {len(puzzles)} puzzles")
            time.sleep(self.delay)
            return puzzles

        except Exception as e:
            print(f"  ✗ Error scraping {url}: {e}", file=sys.stderr)
            return []

    def scrape_multiple(self, urls):
        """
        Scrape puzzles from multiple URLs.

        Args:
            urls (list): List of URLs to scrape

        Returns:
            list: Combined list of all puzzles
        """
        all_puzzles = []

        for url in urls:
            puzzles = self.scrape_url(url)
            all_puzzles.extend(puzzles)

        return all_puzzles


def main():
    """Main function to run the scraper."""
    if len(sys.argv) < 2:
        print("Usage: python scrape_wtharvey.py <url1> [url2 url3 ...] [-o output.json]")
        print("\nExamples:")
        print("  python scrape_wtharvey.py https://wtharvey.com/arkh.html")
        print("  python scrape_wtharvey.py https://wtharvey.com/m8n4.txt")
        print("  python scrape_wtharvey.py arkh.html 1994.html -o puzzles.json")
        print("  python scrape_wtharvey.py m8n2.txt m8n3.txt m8n4.txt -o mate_puzzles.json")
        sys.exit(1)

    # Parse arguments
    urls = []
    output_file = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-o' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            urls.append(sys.argv[i])
            i += 1

    if not urls:
        print("Error: No URLs provided", file=sys.stderr)
        sys.exit(1)

    # Default output file
    if not output_file:
        output_file = "wtharvey_puzzles.json"

    # Create scraper and scrape puzzles
    scraper = WTHarveyPuzzleScraper(delay=1.0)

    print(f"Scraping {len(urls)} URL(s)...\n")
    puzzles = scraper.scrape_multiple(urls)

    # Write output
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(puzzles, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Scraped {len(puzzles)} total puzzles")
    print(f"✓ Output written to: {output_path}")


if __name__ == '__main__':
    main()
