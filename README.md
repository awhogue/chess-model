# Chess Advisor

Fine-tune/RL a model to be good at chess problems. 

## Usage

Process a PNG image containing a board position and send it to Gemini to get a recommended move:

```bash
python extract_board_from_image.py image.png 
```

Train a model:

```bash
python train.py --model-config llama --puzzle-file lichess-poular-250k.json --num-samples 100000 
```

Evaluate that model:

```bash
python eval.py --model-config llama --trained-model-dir models/meta-llama/Llama-3.2-3B-100000-lora-32 --num-problems=32
```