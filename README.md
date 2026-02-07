# Chess Advisor

Fine-tune/RL a model to be good at chess problems.

## Prerequisites

Install Stockfish (required for RLVR training and Stockfish-based evaluation):

- macOS: `brew install stockfish`
- Ubuntu/Debian: `apt install stockfish`
- Other: download from https://stockfishchess.org/download/

## Usage

Process a PNG image containing a board position and send it to Gemini to get a recommended move:

```bash
python extract_board_from_image.py image.png
```

### SFT Training

Train a model with supervised fine-tuning:

```bash
python train.py --model-config llama --puzzle-file data/lichess-popular-250k.json --num-samples 100000
```

### RLVR Training (Stockfish Rewards)

After SFT training, refine the model using GRPO with Stockfish as the reward signal:

```bash
python train_rl.py \
    --model-config llama \
    --sft-adapter-dir models/meta-llama/Llama-3.2-3B-100000-lora-32-epochs-3 \
    --puzzle-file data/lichess-popular-250k.json \
    --num-samples 5000 \
    --max-steps 500
```

Test reward functions before training:

```bash
python train_rl.py \
    --model-config llama \
    --puzzle-file data/wtharvey-sample.json \
    --test-rewards
```

### Evaluate

Evaluate a model (with Stockfish move quality scoring):

```bash
python eval.py \
    --model-config llama \
    --trained-model-dir models/meta-llama/Llama-3.2-3B-5000-rl-lora-64-sft \
    --puzzle-file data/wtharvey-sample.json \
    --num-problems 100
```

Stockfish scoring is automatic if Stockfish is installed. It adds average centipawn loss and optimal move rate metrics alongside the existing accuracy metrics.
