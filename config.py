"""
Model configuration settings for chess puzzle training.
"""

MODEL_CONFIGS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "trust_remote_code": False,
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-3B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         # "gate_proj", "up_proj", "down_proj" Add these to train MLP layers
                         ],
        "trust_remote_code": False,
    },
}

