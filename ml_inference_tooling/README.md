# ML Inference Profiler

A flexible and comprehensive framework for profiling and comparing different ML inference tools, including ONNX Runtime, TorchScript, TensorRT, Triton Inference Server, and vLLM.

## Features

- **Modular Design**: Easily extend to support additional inference frameworks
- **Consistent Benchmarking**: Compare performance across frameworks using the same metrics
- **Comprehensive Metrics**: Measure latency, throughput, memory usage, and model loading time
- **Batch Size Scaling**: Evaluate performance across different batch sizes
- **Visualization**: Generate plots and reports for easy comparison
- **Framework Conversion**: Utilities for converting models between formats

## Installation

```bash
# Clone the repository
git clone https://github.com/username/ml_inference_profiler.git
cd ml_inference_profiler

# Install the base package
pip install -e .

# Install dependencies for specific frameworks
pip install -e ".[onnx]"     # For ONNX Runtime
pip install -e ".[torch]"    # For PyTorch/TorchScript
pip install -e ".[triton]"   # For Triton Inference Server
pip install -e ".[vllm]"     # For vLLM (requires CUDA)
pip install -e ".[all]"      # Install all dependencies except TensorRT
```

Note: TensorRT must be installed separately following NVIDIA's instructions.

## Usage

### Quick Start

To profile a model using all available frameworks:

```bash
# List available models
python examples/profile_all.py --list-models

# Profile ResNet50 using ONNX Runtime, TorchScript, and TensorRT
python examples/profile_all.py --model resnet50 --frameworks onnx torchscript tensorrt --device cuda
```

### Profiling a Computer Vision Model

```bash
# Profile ResNet50
python examples/profile_resnet.py --device cuda --batch-sizes 1 2 4 8 16 32
```

### Profiling an NLP Model

```bash
# Profile BERT
python examples/profile_bert.py --model-name bert-base-uncased --device cuda --batch-sizes 1 2 4 8
```

### Using the Framework in Your Code

```python
from profiler.onnx_profiler import ONNXProfiler
from utils.metrics import MetricsAnalyzer

# Create profiler
profiler = ONNXProfiler(
    model_path="path/to/model.onnx",
    device="cuda",
    batch_sizes=[1, 2, 4, 8],
    warmup_runs=10,
    benchmark_runs=50,
    config={
        "input_shape": [1, 3, 224, 224],
        "input_name": "input"
    }
)

# Run profiling
results = profiler.run_profiling()

# Save and visualize results
profiler.save_results("results/model_onnx_cuda.json")
profiler.plot_results("results")

# Compare with other framework results
analyzer = MetricsAnalyzer("results")
analyzer.plot_latency_comparison("results/latency_comparison.png")
analyzer.plot_throughput_comparison("results/throughput_comparison.png")
analyzer.generate_summary_report("results/summary_report.json")
```

## Supported Frameworks

### ONNX Runtime

Supports both CPU and GPU inference using the ONNX format.

```python
from profiler.onnx_profiler import ONNXProfiler

profiler = ONNXProfiler(
    model_path="model.onnx",
    device="cuda",  # or "cpu"
    batch_sizes=[1, 2, 4, 8, 16],
    config={
        "input_shape": [1, 3, 224, 224],
        "input_name": "input"
    }
)
```

### TorchScript

Supports both CPU and GPU inference using PyTorch's TorchScript format.

```python
from profiler.torchscript_profiler import TorchScriptProfiler

profiler = TorchScriptProfiler(
    model_path="model.pt",
    device="cuda",  # or "cpu"
    batch_sizes=[1, 2, 4, 8, 16],
    config={
        "input_shape": [1, 3, 224, 224]
    }
)
```

### TensorRT

High-performance GPU inference using NVIDIA's TensorRT.

```python
from profiler.tensorrt_profiler import TensorRTProfiler

profiler = TensorRTProfiler(
    model_path="model.onnx",  # or .engine file
    device="cuda",  # TensorRT requires CUDA
    batch_sizes=[1, 2, 4, 8, 16],
    config={
        "input_shape": [1, 3, 224, 224],
        "precision": "fp16"  # "fp32", "fp16", or "int8"
    }
)
```

### Triton Inference Server

Profiling models deployed on Triton Inference Server.

```python
from profiler.triton_profiler import TritonProfiler

profiler = TritonProfiler(
    model_name="resnet50",
    url="localhost:8000",
    device="cuda",  # or "cpu"
    batch_sizes=[1, 2, 4, 8, 16],
    config={
        "input_shape": [1, 3, 224, 224],
        "protocol": "http"  # or "grpc"
    }
)
```

### vLLM

High-throughput LLM inference using vLLM.

```python
from profiler.vllm_profiler import vLLMProfiler

profiler = vLLMProfiler(
    model_path="gpt2",  # HuggingFace model name
    device="cuda",  # vLLM requires CUDA
    batch_sizes=[1, 2, 4, 8],
    config={
        "max_tokens": 128,
        "temperature": 0.0
    }
)
```

## Model Conversion

The framework provides utilities for converting models between formats:

```python
from utils.model_converter import ModelConverter
import torch
import torchvision.models as models

# Load PyTorch model
model = models.resnet50(pretrained=True)
model.eval()

# Convert to ONNX
onnx_path = ModelConverter.pytorch_to_onnx(
    model,
    input_shape=[1, 3, 224, 224],
    output_path="models/resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# Convert to TorchScript
torchscript_path = ModelConverter.pytorch_to_torchscript(
    model,
    input_shape=[1, 3, 224, 224],
    output_path="models/resnet50.pt"
)

# Convert ONNX to TensorRT
tensorrt_path = ModelConverter.onnx_to_tensorrt(
    onnx_path="models/resnet50.onnx",
    output_path="models/resnet50.engine",
    precision="fp16",
    max_batch_size=16
)
```

## Analyzing Results

The `MetricsAnalyzer` class provides tools for comparing results across frameworks:

```python
from utils.metrics import MetricsAnalyzer

analyzer = MetricsAnalyzer("results_directory")

# Generate comparison plots
analyzer.plot_latency_comparison("plots/latency_comparison.png")
analyzer.plot_throughput_comparison("plots/throughput_comparison.png")
analyzer.plot_memory_comparison("plots/memory_comparison.png")
analyzer.plot_load_time_comparison("plots/load_time_comparison.png")
analyzer.plot_scaling_efficiency("plots/scaling_efficiency.png")

# Generate summary report
summary = analyzer.generate_summary_report("reports/summary_report.json")
```

## Directory Structure

```
ml_inference_profiler/
│
├── profiler/              # Profiler implementations
│   ├── base_profiler.py   # Base class with common profiling logic
│   ├── onnx_profiler.py   # ONNX Runtime profiler
│   ├── torchscript_profiler.py
│   ├── triton_profiler.py
│   ├── vllm_profiler.py
│   └── tensorrt_profiler.py
│
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading utilities
│   ├── model_converter.py # Model format conversion
│   ├── metrics.py         # Metrics collection and visualization
│   └── device_info.py     # System information
│
├── config/                # Configuration
│   ├── default_config.py  # Default settings
│   └── models_config.py   # Model-specific configs
│
├── examples/              # Example scripts
│   ├── profile_all.py     # Profile with all frameworks
│   ├── profile_resnet.py  # CV model example
│   └── profile_bert.py    # NLP model example
│
├── requirements.txt       # Dependencies
├── setup.py               # Package setup
└── README.md              # Documentation
```

## Requirements

- Python 3.7+
- NumPy, Pandas, Matplotlib, Seaborn
- Framework-specific dependencies:
  - ONNX Runtime: `onnxruntime` or `onnxruntime-gpu`
  - TorchScript: `torch`
  - Triton: `tritonclient[all]`
  - vLLM: `vllm`
  - TensorRT: NVIDIA TensorRT installation

## License

MIT License