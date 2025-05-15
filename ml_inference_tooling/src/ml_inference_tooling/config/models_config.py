"""
Configuration for specific models and frameworks.
"""

# Import base configurations
from .default_config import DEFAULT_CONFIG, get_model_config, merge_configs

# Framework-specific model configurations
MODELS_CONFIG = {
    # Computer Vision Models
    "resnet50": {
        "description": "ResNet-50 classification model",
        "onnx": {
            "model_url": "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx",
            "input_shape": [1, 3, 224, 224],
            "input_name": "data",
            "output_name": "resnetv24_dense0_fwd",
            "datatype": "FP32"
        },
        "torchscript": {
            "model_class": "torchvision.models.resnet50",
            "model_url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
            "input_shape": [1, 3, 224, 224]
        },
        "tensorrt": {
            "precision": "fp16",
            "workspace_size": 1 << 30,
            "input_shape": [1, 3, 224, 224]
        }
    },
    
    "mobilenet_v2": {
        "description": "MobileNet V2 classification model",
        "onnx": {
            "model_url": "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
            "input_shape": [1, 3, 224, 224],
            "input_name": "data",
            "output_name": "mobilenetv20_output_flatten0_reshape0",
            "datatype": "FP32"
        },
        "torchscript": {
            "model_class": "torchvision.models.mobilenet_v2",
            "model_url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "input_shape": [1, 3, 224, 224]
        },
        "tensorrt": {
            "precision": "fp16",
            "workspace_size": 1 << 30,
            "input_shape": [1, 3, 224, 224]
        }
    },
    
    "yolov5s": {
        "description": "YOLOv5 small object detection model",
        "onnx": {
            "model_url": "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.onnx",
            "input_shape": [1, 3, 640, 640],
            "input_name": "images",
            "output_name": "output",
            "datatype": "FP32"
        },
        "torchscript": {
            "model_url": "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.torchscript",
            "input_shape": [1, 3, 640, 640]
        },
        "tensorrt": {
            "precision": "fp16",
            "workspace_size": 1 << 30,
            "input_shape": [1, 3, 640, 640]
        }
    },
    
    # NLP Models
    "bert-base-uncased": {
        "description": "BERT base uncased NLP model",
        "onnx": {
            "model_url": "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
            "input_shape": [1, 128],  # [batch_size, sequence_length]
            "input_names": ["input_ids", "input_mask", "segment_ids"],
            "output_names": ["output_start_logits", "output_end_logits"],
            "datatype": "INT64"
        },
        "vllm": {
            "model_name": "bert-base-uncased",
            "max_tokens": 64,
            "temperature": 0.0
        }
    },
    
    "gpt2": {
        "description": "GPT-2 language model",
        "onnx": {
            "model_path": None,  # No pre-built ONNX available, would need conversion
            "input_shape": [1, 128],
            "input_name": "input_ids",
            "output_name": "logits",
            "datatype": "INT64"
        },
        "vllm": {
            "model_name": "gpt2",
            "max_tokens": 128,
            "temperature": 0.7,
            "top_p": 0.9
        }
    },
    
    "t5-small": {
        "description": "T5 small text-to-text model",
        "vllm": {
            "model_name": "t5-small",
            "max_tokens": 128,
            "temperature": 0.0
        }
    }
}

# Examples for popular models
def get_benchmarking_models():
    """Get a list of recommended models for benchmarking."""
    return {
        "cv": ["resnet50", "mobilenet_v2", "yolov5s"],
        "nlp": ["bert-base-uncased", "gpt2", "t5-small"]
    }

def get_full_config(model_name, framework):
    """
    Get the complete configuration for a specific model and framework.
    
    Args:
        model_name: Name of the model
        framework: Name of the framework
        
    Returns:
        Merged configuration
    """
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model '{model_name}' not found in configuration")
        
    model_config = MODELS_CONFIG[model_name]
    
    if framework not in model_config:
        raise ValueError(f"Framework '{framework}' not found for model '{model_name}'")
        
    framework_config = model_config[framework]
    
    # Merge configs
    base_config = DEFAULT_CONFIG.copy()
    if framework in base_config:
        base_framework_config = base_config[framework].copy()
        framework_config = {**base_framework_config, **framework_config}
    
    return {"model_name": model_name, "framework": framework, **framework_config}