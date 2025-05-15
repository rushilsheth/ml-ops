"""
Examples - Usage examples for the GPU simulator
"""
import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from gpu_simulator import GPUSimulator
from tensor_ops import Tensor, calculate_tensor_size
from memory_manager import MemoryManager
from multi_gpu import MultiGPUManager
from load_tester import LoadTester, MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gpu_simulator.examples')


def basic_tensor_operations():
    """Example of basic tensor operations with a single GPU"""
    print("\n=== Basic Tensor Operations ===")
    
    # Initialize a simulated GPU
    gpu = GPUSimulator(memory_size=16 * 1024)  # 16GB GPU
    print(f"Created GPU simulator: {gpu}")
    
    # Create some tensors
    input_tensor = Tensor(shape=(128, 768), name="input_embeddings", gpu=gpu)
    weight_tensor = Tensor(shape=(768, 3072), name="attention_weights", gpu=gpu)
    
    print(f"Created tensors:")
    print(f"  {input_tensor}")
    print(f"  {weight_tensor}")
    
    # Perform matrix multiplication
    print("\nPerforming matrix multiplication...")
    start_time = time.time()
    result = input_tensor @ weight_tensor  # Using the @ operator
    end_time = time.time()
    
    print(f"Result tensor: {result}")
    print(f"Operation took {(end_time - start_time) * 1000:.2f}ms")
    
    # Check memory usage
    memory_info = gpu.memory_usage()
    print("\nGPU Memory Usage:")
    print(f"  Total memory: {memory_info['total_memory']}MB")
    print(f"  Allocated memory: {memory_info['allocated_memory']:.2f}MB")
    print(f"  Free memory: {memory_info['free_memory']:.2f}MB")
    print(f"  Utilization: {memory_info['utilization']:.2f}%")
    
    # Perform more operations
    print("\nPerforming activation functions...")
    relu_result = result.relu()
    sigmoid_result = result.sigmoid()
    
    print(f"ReLU result: {relu_result}")
    print(f"Sigmoid result: {sigmoid_result}")
    
    # Check performance stats
    stats = gpu.get_performance_stats()
    print("\nPerformance Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Operation counts: {stats['operation_counts']}")
    print(f"  Total estimated FLOPS: {stats['total_flops']:.2e}")
    print(f"  Total time: {stats['total_time']:.6f}s")
    print(f"  Effective TFLOPS: {stats['effective_tflops']:.2f}")


def self_attention_example():
    """Example of simulating a self-attention operation"""
    print("\n=== Self-Attention Simulation ===")
    
    # Initialize a GPU simulator
    gpu = GPUSimulator(memory_size=24 * 1024)  # 24GB GPU
    print(f"Created GPU simulator: {gpu}")
    
    # Create attention tensors
    batch_size = 32
    seq_len = 512
    d_model = 1024
    
    # Query, Key, Value tensors
    query = Tensor(shape=(batch_size, seq_len, d_model), name="query", gpu=gpu)
    key = Tensor(shape=(batch_size, seq_len, d_model), name="key", gpu=gpu)
    value = Tensor(shape=(batch_size, seq_len, d_model), name="value", gpu=gpu)
    
    print(f"Created attention tensors:")
    print(f"  {query}")
    print(f"  {key}")
    print(f"  {value}")
    
    # Perform attention operation
    print("\nPerforming attention operation...")
    start_time = time.time()
    attn_output = gpu.attention(query, key, value)
    end_time = time.time()
    
    print(f"Attention output: {attn_output}")
    print(f"Operation took {(end_time - start_time) * 1000:.2f}ms")
    
    # Check GPU stats
    memory_info = gpu.memory_usage()
    print("\nGPU Memory Usage:")
    print(f"  Total memory: {memory_info['total_memory']}MB")
    print(f"  Allocated memory: {memory_info['allocated_memory']:.2f}MB")
    print(f"  Free memory: {memory_info['free_memory']:.2f}MB")
    print(f"  Utilization: {memory_info['utilization']:.2f}%")


def multi_gpu_example():
    """Example of multi-GPU operations"""
    print("\n=== Multi-GPU Operations ===")
    
    # Initialize multi-GPU environment
    num_gpus = 4
    gpu_manager = MultiGPUManager(num_gpus=num_gpus, memory_per_gpu=16 * 1024)
    print(f"Created multi-GPU environment: {gpu_manager}")
    
    # Get individual GPUs
    gpus = [gpu_manager.get_gpu(i) for i in range(num_gpus)]
    
    # Create a tensor on the first GPU
    tensor_shape = (64, 1024)
    tensor_on_gpu0 = Tensor(shape=tensor_shape, name="tensor_gpu0", gpu=gpus[0])
    print(f"Created tensor on GPU 0: {tensor_on_gpu0}")
    
    # Broadcast to all GPUs
    print("\nBroadcasting tensor to all GPUs...")
    start_time = time.time()
    tensor_copies = gpu_manager.broadcast(tensor_on_gpu0, src_gpu_id=0)
    end_time = time.time()
    
    print(f"Broadcast complete in {(end_time - start_time) * 1000:.2f}ms")
    for gpu_id, tensor in tensor_copies.items():
        print(f"  GPU {gpu_id}: {tensor}")
    
    # Perform operations on each GPU
    results = {}
    print("\nPerforming operations on each GPU...")
    for gpu_id, tensor in tensor_copies.items():
        # Create a weight tensor on this GPU
        weight = Tensor(shape=(1024, 2048), name=f"weight_gpu{gpu_id}", gpu=gpus[gpu_id])
        
        # Perform matmul
        results[gpu_id] = tensor @ weight
        print(f"  GPU {gpu_id} result: {results[gpu_id]}")
    
    # All-reduce to combine results
    print("\nPerforming all-reduce to combine results...")
    start_time = time.time()
    combined_result = gpu_manager.all_reduce(results[0])  # Using first GPU's result
    end_time = time.time()
    
    print(f"All-reduce complete in {(end_time - start_time) * 1000:.2f}ms")
    print(f"Combined result: {combined_result}")
    
    # Check memory usage across GPUs
    memory_stats = gpu_manager.get_memory_stats()
    print("\nMemory Usage Across GPUs:")
    for gpu_id, stats in memory_stats.items():
        if gpu_id != "aggregate":
            print(f"  GPU {gpu_id}: {stats['allocated_memory']:.2f}MB / {stats['total_memory']}MB "
                 f"({stats['utilization']:.2f}%)")
    
    print(f"  Aggregate: {memory_stats['aggregate']['allocated_memory']:.2f}MB / "
          f"{memory_stats['aggregate']['total_memory']}MB "
          f"({memory_stats['aggregate']['utilization']:.2f}%)")


def transformer_layer_example():
    """Example simulating a complete transformer layer"""
    print("\n=== Transformer Layer Simulation ===")
    
    # Initialize GPU
    gpu = GPUSimulator(memory_size=32 * 1024)  # 32GB GPU
    print(f"Created GPU simulator: {gpu}")
    
    # Configuration
    batch_size = 16
    seq_len = 1024
    d_model = 1024
    d_ff = 4096
    num_heads = 16
    head_dim = d_model // num_heads
    
    # Input tensor
    input_tensor = Tensor(shape=(batch_size, seq_len, d_model), name="layer_input", gpu=gpu)
    print(f"Created input tensor: {input_tensor}")
    
    # Transformer layer forward pass
    print("\nPerforming transformer layer forward pass...")
    
    # 1. Multi-head self-attention
    # Project query, key, value
    print("  1. Computing QKV projections...")
    qkv_weights = Tensor(shape=(d_model, 3 * d_model), name="qkv_weights", gpu=gpu)
    qkv_projections = input_tensor @ qkv_weights
    
    # Split into query, key, value
    query = Tensor(shape=(batch_size, seq_len, d_model), name="query", gpu=gpu)
    key = Tensor(shape=(batch_size, seq_len, d_model), name="key", gpu=gpu)
    value = Tensor(shape=(batch_size, seq_len, d_model), name="value", gpu=gpu)
    
    # Self-attention
    print("  2. Computing self-attention...")
    attn_output = gpu.attention(query, key, value)
    
    # Project attention output
    print("  3. Computing attention output projection...")
    attn_output_weights = Tensor(shape=(d_model, d_model), name="attn_output_weights", gpu=gpu)
    attn_output_projected = attn_output @ attn_output_weights
    
    # Residual connection and layer norm
    print("  4. Adding residual connection and layer norm...")
    residual1 = Tensor(shape=(batch_size, seq_len, d_model), name="residual1", gpu=gpu)
    norm1 = residual1.layernorm()
    
    # Feed-forward network
    print("  5. Computing feed-forward network...")
    ff1_weights = Tensor(shape=(d_model, d_ff), name="ff1_weights", gpu=gpu)
    ff1_output = norm1 @ ff1_weights
    ff1_activated = ff1_output.relu()
    
    ff2_weights = Tensor(shape=(d_ff, d_model), name="ff2_weights", gpu=gpu)
    ff2_output = ff1_activated @ ff2_weights
    
    # Final residual connection and layer norm
    print("  6. Adding final residual connection and layer norm...")
    residual2 = Tensor(shape=(batch_size, seq_len, d_model), name="residual2", gpu=gpu)
    output = residual2.layernorm()
    
    print(f"Transformer layer output: {output}")
    
    # Check memory and performance stats
    memory_info = gpu.memory_usage()
    print("\nGPU Memory Usage:")
    print(f"  Total memory: {memory_info['total_memory']}MB")
    print(f"  Allocated memory: {memory_info['allocated_memory']:.2f}MB")
    print(f"  Free memory: {memory_info['free_memory']:.2f}MB")
    print(f"  Utilization: {memory_info['utilization']:.2f}%")
    
    stats = gpu.get_performance_stats()
    print("\nPerformance Statistics:")
    print(f"  Total operations: {stats['total_operations']}")
    print(f"  Operation counts: {stats['operation_counts']}")
    print(f"  Effective TFLOPS: {stats['effective_tflops']:.2f}")


def load_testing_example():
    """Example of load testing"""
    print("\n=== Load Testing ===")
    
    # Initialize the load tester with 4 GPUs
    tester = LoadTester(num_gpus=4, memory_per_gpu=24 * 1024)  # 4x 24GB GPUs
    print(f"Created load tester with 4 GPUs")
    
    # Calculate memory requirements for different models
    models = ["1B", "7B", "13B"]
    for model in models:
        mem_req = tester.calculate_model_memory(model_size=model, precision="float16")
        print(f"\nMemory requirements for {model} model (float16):")
        print(f"  Model parameters: {mem_req['total_parameters']:,}")
        print(f"  Model memory: {mem_req['model_memory_mb']:.2f}MB")
        print(f"  Inference memory: {mem_req['inference_memory_mb']:.2f}MB")
        print(f"  Training memory: {mem_req['training_memory_mb']:.2f}MB")
        print(f"  Training with checkpointing: {mem_req['training_checkpointed_mb']:.2f}MB")
    
    # Run inference test for 7B model
    print("\nRunning inference test for 7B model...")
    inference_results = tester.run_inference_test(
        batch_sizes=[1, 2, 4, 8, 16],
        model_size="7B",
        precision="float16",
        sequence_length=2048
    )
    
    # Plot the results
    print("\nPlotting inference test results...")
    try:
        tester.plot_results(inference_results, plot_type="throughput")
        tester.plot_results(inference_results, plot_type="memory")
        tester.plot_results(inference_results, plot_type="latency")
    except Exception as e:
        print(f"Could not plot results: {str(e)}")
    
    # Run training test for 1B model
    print("\nRunning training test for 1B model...")
    training_results = tester.run_training_test(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        model_size="1B",
        precision="float16",
        sequence_length=1024,
        use_grad_checkpointing=True
    )
    
    # Export the results
    print("\nExporting test results...")
    try:
        tester.export_results(inference_results, filename="inference_results.json")
        tester.export_results(training_results, filename="training_results.json")
    except Exception as e:
        print(f"Could not export results: {str(e)}")


def memory_management_example():
    """Example demonstrating memory management features"""
    print("\n=== Memory Management ===")
    
    # Initialize memory manager
    memory_size = 16 * 1024  # 16GB
    memory_manager = MemoryManager(total_memory=memory_size)
    print(f"Created memory manager with {memory_size}MB")
    
    # Allocate some blocks
    print("\nAllocating memory blocks...")
    blocks = []
    try:
        blocks.append(memory_manager.allocate(1024, "large_block1"))
        blocks.append(memory_manager.allocate(2048, "large_block2"))
        blocks.append(memory_manager.allocate(512, "medium_block1"))
        blocks.append(memory_manager.allocate(256, "medium_block2"))
        
        for i in range(10):
            blocks.append(memory_manager.allocate(128, f"small_block{i+1}"))
    except MemoryError as e:
        print(f"Memory allocation failed: {str(e)}")
    
    # Print memory info
    memory_info = memory_manager.memory_info()
    print("\nMemory Usage:")
    print(f"  Total memory: {memory_info['total_memory']}MB")
    print(f"  Allocated memory: {memory_info['allocated_memory']}MB")
    print(f"  Free memory: {memory_info['free_memory']}MB")
    print(f"  Utilization: {memory_info['utilization']:.2f}%")
    print(f"  Fragmentation: {memory_info['fragmentation']:.4f}")
    
    # Free some blocks to create fragmentation
    print("\nFreeing some blocks to create fragmentation...")
    for i in range(0, len(blocks), 2):
        if i < len(blocks):
            memory_manager.free(blocks[i])
    
    # Print updated memory info
    memory_info = memory_manager.memory_info()
    print("\nMemory Usage After Freeing Blocks:")
    print(f"  Total memory: {memory_info['total_memory']}MB")
    print(f"  Allocated memory: {memory_info['allocated_memory']}MB")
    print(f"  Free memory: {memory_info['free_memory']}MB")
    print(f"  Utilization: {memory_info['utilization']:.2f}%")
    print(f"  Fragmentation: {memory_info['fragmentation']:.4f}")
    print(f"  Number of free blocks: {memory_info['num_free_blocks']}")
    
    # Try to allocate a large block that requires compaction
    print("\nTrying to allocate a large block requiring compaction...")
    try:
        large_addr = memory_manager.allocate(4096, "very_large_block")
        print(f"  Successfully allocated 4096MB block at address {large_addr}")
    except MemoryError as e:
        print(f"  Memory allocation failed: {str(e)}")
    
    # Final memory info
    memory_info = memory_manager.memory_info()
    print("\nFinal Memory Info:")
    print(f"  Allocation by size: {memory_info['allocation_by_size']}")
    print(f"  Peak memory usage: {memory_info['peak_memory_usage']}MB")
    print(f"  Number of allocations: {memory_info['num_allocations']}")
    print(f"  Number of deallocations: {memory_info['num_deallocations']}")


def model_parallelism_example():
    """Example demonstrating model parallelism"""
    print("\n=== Model Parallelism Example ===")
    
    # Initialize multi-GPU environment
    num_gpus = 4
    gpu_manager = MultiGPUManager(num_gpus=num_gpus, memory_per_gpu=16 * 1024)
    print(f"Created multi-GPU environment with {num_gpus} GPUs")
    
    # Configure model parallelism
    gpu_manager.setup_model_parallel(group_size=num_gpus)
    
    # Model configuration (using 70B model split across 4 GPUs)
    config = MODEL_CONFIGS["70B"]
    batch_size = 1
    seq_len = 2048
    
    # Calculate layers per GPU
    layers_per_gpu = config["num_layers"] // num_gpus
    print(f"\nSplitting {config['num_layers']} transformer layers across {num_gpus} GPUs "
          f"({layers_per_gpu} layers per GPU)")
    
    # Input tensor on first GPU
    input_shape = (batch_size, seq_len, config["hidden_size"])
    input_tensor = Tensor(shape=input_shape, name="input_embeddings", gpu=gpu_manager.gpus[0])
    print(f"Created input tensor on GPU 0: {input_tensor}")
    
    # Process layers in pipeline fashion
    hidden_states = input_tensor
    current_gpu = 0
    
    print("\nProcessing layers in pipeline fashion...")
    for layer_idx in range(min(8, config["num_layers"])):  # Simulate only a few layers
        gpu = gpu_manager.gpus[current_gpu]
        
        print(f"  Layer {layer_idx} on GPU {current_gpu}...")
        
        # Self-attention
        query = Tensor(shape=hidden_states.shape, name=f"query_{layer_idx}", gpu=gpu)
        key = Tensor(shape=hidden_states.shape, name=f"key_{layer_idx}", gpu=gpu)
        value = Tensor(shape=hidden_states.shape, name=f"value_{layer_idx}", gpu=gpu)
        attn_output = gpu.attention(query, key, value)
        
        # FFN
        ffn_intermediate_shape = (batch_size, seq_len, config["ffn_hidden_size"])
        ffn_intermediate = Tensor(shape=ffn_intermediate_shape, name=f"ffn_intermediate_{layer_idx}", gpu=gpu)
        ffn_output = Tensor(shape=hidden_states.shape, name=f"ffn_output_{layer_idx}", gpu=gpu)
        
        # Update hidden states
        hidden_states = ffn_output
        
        # Move to next GPU if needed
        if (layer_idx + 1) % layers_per_gpu == 0 and current_gpu < num_gpus - 1:
            current_gpu += 1
            print(f"  Transferring output to GPU {current_gpu}...")
            hidden_states = hidden_states.to_gpu(gpu_manager.gpus[current_gpu])
    
    # Final output on last active GPU
    output_tensor = Tensor(
        shape=(batch_size, seq_len, config["vocab_size"]),
        name="output_logits",
        gpu=gpu_manager.gpus[current_gpu]
    )
    print(f"\nFinal output on GPU {current_gpu}: {output_tensor}")
    
    # Check memory usage across GPUs
    memory_stats = gpu_manager.get_memory_stats()
    print("\nMemory Usage Across GPUs:")
    for gpu_id, stats in memory_stats.items():
        if gpu_id != "aggregate":
            print(f"  GPU {gpu_id}: {stats['allocated_memory']:.2f}MB / {stats['total_memory']}MB "
                 f"({stats['utilization']:.2f}%)")


def main():
    """Run all examples"""
    print("GPU Simulator Examples")
    print("=====================")
    
    # Run examples
    try:
        basic_tensor_operations()
    except Exception as e:
        print(f"Error in basic tensor operations example: {str(e)}")
    
    try:
        self_attention_example()
    except Exception as e:
        print(f"Error in self-attention example: {str(e)}")
    
    try:
        multi_gpu_example()
    except Exception as e:
        print(f"Error in multi-GPU example: {str(e)}")
    
    try:
        transformer_layer_example()
    except Exception as e:
        print(f"Error in transformer layer example: {str(e)}")
    
    try:
        load_testing_example()
    except Exception as e:
        print(f"Error in load testing example: {str(e)}")
    
    try:
        memory_management_example()
    except Exception as e:
        print(f"Error in memory management example: {str(e)}")
    
    try:
        model_parallelism_example()
    except Exception as e:
        print(f"Error in model parallelism example: {str(e)}")
    
    print("\nAll examples completed.")


if __name__ == "__main__":
    main()