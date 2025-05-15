# GPU Simulator

This package provides a simulation environment for GPU operations focused on language model training and inference. It allows for testing and debugging LLM workflows without requiring physical GPU hardware.

## Features

- Simulate CUDA operations and memory management
- Track tensor operations and memory usage
- Simulate multi-GPU environments
- Configurable performance characteristics
- Load testing capabilities for inference and training

## Files

- `gpu_simulator.py` - Core simulation engine
- `tensor_ops.py` - Tensor operation simulation
- `memory_manager.py` - GPU memory simulation
- `multi_gpu.py` - Multi-GPU simulation
- `load_tester.py` - Load testing utilities
- `examples.py` - Usage examples

## Usage

```python
from gpu_simulator import GPUSimulator
from tensor_ops import Tensor

# Initialize a simulated GPU
gpu = GPUSimulator(memory_size=12 * 1024)  # 12GB GPU

# Create tensors
input_tensor = Tensor(shape=(512, 768), name="input_embeddings")
weight_tensor = Tensor(shape=(768, 2048), name="attention_weights")

# Perform operations
result = gpu.matmul(input_tensor, weight_tensor)

# Check memory usage
print(gpu.memory_usage())
```

## Load Testing

The load testing module allows you to simulate various workloads:

```python
from load_tester import LoadTester
from gpu_simulator import GPUSimulator

# Create a load test with 4 simulated GPUs
tester = LoadTester(num_gpus=4)

# Run a batch inference test
results = tester.run_inference_test(
    batch_sizes=[1, 2, 4, 8, 16, 32],
    model_size="7B",
    sequence_length=2048
)

# Visualize results
tester.plot_results(results)
```

## Limitations

This simulator approximates GPU behavior but is not cycle-accurate. It provides reasonable estimates for memory usage, operation time, and throughput, but should not be used for precise performance benchmarking.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (for visualization)