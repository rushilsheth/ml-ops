# ML-Ops Toolkit

A comprehensive collection of tools and frameworks for training, deploying, monitoring, and optimizing ML models at scale.

## Overview

This repository contains modular components that address various aspects of the machine learning lifecycle, from development to production deployment. The toolkit focuses on scalability, performance optimization, and standardization of ML workflows.

## Key Components

### 🚀 Distributed Training Infrastructure
- GPU simulation framework for testing parallelized training
- Resources for distributed data loading and preprocessing
- Tools for monitoring cluster utilization

### 📊 Model Deployment & Inference
- ML inference profiling framework supporting ONNX, TorchScript, TensorRT, Triton, and vLLM
- Batch processing optimization tools
- Latency and throughput benchmarking utilities

### 🔍 Monitoring & Observability
- Model performance tracking
- Data drift detection
- Resource utilization monitoring

## Project Structure

```
ml-ops/
├── src/
│   ├── gpu_simulator/       # Distributed GPU simulation tools
│   │   ├── __init__.py
│   │   └── main.py          # Main application
│   │
│   ├── inference_profiler/  # Framework for benchmarking inference tools
│   │   ├── profiler/        # Profiler implementations
│   │   ├── utils/           # Utility functions
│   │   ├── config/          # Configuration files
│   │   └── examples/        # Example scripts
│   │
│   └── monitoring/          # Monitoring tools
│       └── ...
│
├── scripts/                 # Utility scripts
│   └── install.sh           # Installation script
│
├── notebooks/               # Jupyter notebooks with examples
│   └── ...
│
├── docs/                    # Documentation
│   └── ...
│
├── .env                     # Environment variables
├── pyproject.toml           # Poetry configuration
└── README.md                # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-ops.git
cd ml-ops

# Option 1: Using the installation script
./scripts/install.sh

# Option 2: Using Poetry
poetry install

# Install additional dependencies
poetry install -E gpu      # GPU support
poetry install -E inference  # Inference tools
poetry install -E monitoring # Monitoring tools
```

## Usage Examples

### GPU Simulation

```python
from ml_ops.gpu_simulator import GPUCluster

# Create a simulated cluster with 8 GPUs
cluster = GPUCluster(num_gpus=8, memory_per_gpu=16)

# Run a distributed training simulation
cluster.simulate_training(
    model_size_gb=40,
    batch_size=32,
    parallelism_strategy="data_parallel"
)
```

### Inference Profiling

```bash
# Profile ResNet50 using multiple inference frameworks
python -m ml_ops.inference_profiler.examples.profile_all --model resnet50 --frameworks onnx torchscript tensorrt
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.