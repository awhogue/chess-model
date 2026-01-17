sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "torch>=2.5.0" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

source .api_keys

wandb login
mkdir models
