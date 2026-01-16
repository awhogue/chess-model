"""
Model configuration settings for chess puzzle training.
"""

MODEL_CONFIGS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "llama8": {
        "name": "meta-llama/Llama-3.2-8B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-3B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj" # Add these to train MLP layers
                         ],
    },
    "qwen14": {
        "name": "Qwen/Qwen2.5-14B",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj" # Add these to train MLP layers
                         ],
    },
}

