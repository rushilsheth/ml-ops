"""
Default configuration for ML inference profiling.
"""

# Common configuration
DEFAULT_CONFIG = {
    # General settings
    "output_dir": "./profiling_results",
    "batch_sizes": [1, 2, 4, 8, 16, 32],
    "warmup_runs": 5,
    "benchmark_runs": 50,
    
    # Device settings
    "use_gpu": True,
    "gpu_device_id": 0,
    
    # Model settings
    "model_type": "classification",  # 'classification', 'detection', 'segmentation', 'nlp', etc.
    
    # Framework-specific settings
    "onnx": {
        "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "enable_profiling": False
    },
    
    "torchscript": {
        "use_jit_fuser": True,
        "use_amp": False  # Automatic Mixed Precision
    },
    
    "triton": {
        "url": "localhost:8000",
        "protocol": "http",  # 'http' or 'grpc'
        "model_name": "model",
        "verbose": False
    },
    
    "vllm": {
        "max_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "block_size": 16,  # KV cache block size
        "tensor_parallel_size": 1  # Number of GPUs for tensor parallelism
    },
    
    "tensorrt": {
        "precision": "fp16",  # 'fp32', 'fp16', or 'int8'
        "workspace_size": 1 << 30,  # 1GB
        "dla_core": -1,  # -1 to disable DLA
        "builder_optimization_level": 3
    }
}

# Model-specific configurations
RESNET50_CONFIG = {
    "input_shape": [1, 3, 224, 224],
    "input_name": "input",
    "output_name": "output",
    "datatype": "FP32"
}

BERT_CONFIG = {
    "input_shape": [1, 128],  # [batch_size, sequence_length]
    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
    "output_name": "output",
    "datatype": "INT64"
}

YOLO_CONFIG = {
    "input_shape": [1, 3, 640, 640],
    "input_name": "images",
    "output_names": ["output"],
    "datatype": "FP32",
    "nms_threshold": 0.45,
    "confidence_threshold": 0.25
}

GPT2_CONFIG = {
    "input_shape": [1, 128],  # [batch_size, sequence_length]
    "input_name": "input_ids",
    "output_name": "logits",
    "datatype": "INT64",
    "max_tokens": 64
}

# Configuration functions
def get_model_config(model_name):
    """Get configuration for a specific model."""
    model_configs = {
        "resnet50": RESNET50_CONFIG,
        "bert-base-uncased": BERT_CONFIG,
        "yolo": YOLO_CONFIG,
        "gpt2": GPT2_CONFIG
    }
    
    return model_configs.get(model_name, {})

def merge_configs(base_config, model_config):
    """Merge base and model-specific configs."""
    import copy
    config = copy.deepcopy(base_config)
    
    # Recursively update config
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    return update_dict(config, model_config)