# Chess Advisor

Fine-tune/RL a model to be good at chess problems. 

## Usage

Process a PNG image containing a board position and send it to Gemini to get a recommended move:

```bash
python extract_board_from_image.py image.png 
```

Run puzzles against a model hosted on openrouter:

```bash
python eval_puzzles.py data/wtharvey-sample.json meta-llama/Llama-3.1-8B-Instruct --num-problems 5
```

...or hosted locally e.g. with Ollama:

```bash
python eval_puzzles.py data/wtharvey-sample.json llama3.1 --local --num-problems 5
```

Coming soon: `train.py`.