from gpu_simulator import GPUSimulator
from tensor_ops import Tensor

def quick_demo():
    # Initialize a simulated GPU
    gpu = GPUSimulator(memory_size=12 * 1024)  # 12GB GPU
    print(f"Initialized GPU with 12GB memory")
    
    # Create tensors
    input_tensor = Tensor(shape=(512, 768), name="input_embeddings", gpu=gpu)
    weight_tensor = Tensor(shape=(768, 2048), name="attention_weights", gpu=gpu)
    
    # Perform matrix multiplication
    print("Performing matrix multiplication...")
    result = input_tensor @ weight_tensor
    print(f"Result shape: {result.shape}")
    
    # Check memory usage
    memory_info = gpu.memory_usage()
    print(f"Memory usage: {memory_info['allocated_memory']:.2f}MB / {memory_info['total_memory']}MB ({memory_info['utilization']:.2f}%)")
    
    # Get performance stats
    stats = gpu.get_performance_stats()
    print(f"Effective performance: {stats['effective_tflops']:.2f} TFLOPS")

if __name__ == "__main__":
    quick_demo()