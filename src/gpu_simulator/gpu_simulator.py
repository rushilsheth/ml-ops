"""
GPU Simulator - Core simulation engine for GPU operations
"""
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gpu_simulator')

class GPUSimulator:
    """Simulates a GPU for language model operations"""
    
    def __init__(self, 
                 device_id: int = 0,
                 memory_size: int = 16 * 1024,  # 16GB in MB
                 compute_capability: float = 8.0,
                 bandwidth: int = 900,  # GB/s for high-end GPU
                 latency: float = 0.001,  # seconds
                 flops: int = 40 * 10**12):  # 40 TFLOPS for a high-end GPU
        """
        Initialize a GPU simulator with specified characteristics
        
        Args:
            device_id: GPU device ID
            memory_size: Memory size in MB
            compute_capability: CUDA compute capability
            bandwidth: Memory bandwidth in GB/s
            latency: Operation latency in seconds
            flops: Floating point operations per second
        """
        self.device_id = device_id
        self.memory_size = memory_size
        self.compute_capability = compute_capability
        self.bandwidth = bandwidth
        self.latency = latency
        self.flops = flops
        
        # Memory tracking
        self.allocated_memory = 0
        self.memory_blocks: Dict[int, Tuple[int, str]] = {}  # addr -> (size, name)
        self.next_addr = 1000  # Starting address
        
        # Operation tracking
        self.op_history: List[Dict] = []
        
        logger.info(f"Initialized GPU {device_id} with {memory_size}MB memory")
    
    def allocate(self, size_mb: float, tensor_name: str = "") -> int:
        """Simulate memory allocation on GPU"""
        if self.allocated_memory + size_mb > self.memory_size:
            raise MemoryError(f"Out of memory: Tried to allocate {size_mb}MB but only "
                              f"{self.memory_size - self.allocated_memory}MB available")
        
        addr = self.next_addr
        self.memory_blocks[addr] = (size_mb, tensor_name)
        self.allocated_memory += size_mb
        self.next_addr += int(size_mb * 1024)  # Increment by bytes
        
        # Add some realistic delay
        time.sleep(self.latency * 0.1 * (size_mb / 1024))
        
        logger.debug(f"Allocated {size_mb}MB for '{tensor_name}' at address {addr}")
        return addr
    
    def free(self, addr: int) -> None:
        """Simulate memory deallocation from GPU"""
        if addr not in self.memory_blocks:
            raise ValueError(f"Invalid address: {addr}")
        
        size_mb, name = self.memory_blocks[addr]
        self.allocated_memory -= size_mb
        del self.memory_blocks[addr]
        
        logger.debug(f"Freed {size_mb}MB from '{name}' at address {addr}")
    
    def memory_usage(self) -> Dict:
        """Return current memory usage statistics"""
        return {
            "total_memory": self.memory_size,
            "allocated_memory": self.allocated_memory,
            "free_memory": self.memory_size - self.allocated_memory,
            "utilization": (self.allocated_memory / self.memory_size) * 100,
            "blocks": len(self.memory_blocks)
        }
    
    def matmul(self, a, b) -> 'Tensor':
        """Simulate matrix multiplication on GPU"""
        from tensor_ops import Tensor  # Import here to avoid circular imports
        
        # Validate tensor dimensions
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible dimensions for matmul: {a.shape} and {b.shape}")
        
        # Calculate output shape
        output_shape = (a.shape[0], b.shape[1])
        
        # Calculate estimated FLOPS for this operation
        # Each element requires n multiplications and n-1 additions
        flops_per_element = a.shape[1] * 2 - 1
        total_flops = output_shape[0] * output_shape[1] * flops_per_element
        
        # Estimate time based on FLOPS and some overhead
        estimated_time = (total_flops / self.flops) * (1.1 + 0.1 * np.random.random())
        
        # Record operation
        op_record = {
            "op": "matmul",
            "input_shapes": [a.shape, b.shape],
            "output_shape": output_shape,
            "flops": total_flops,
            "estimated_time": estimated_time
        }
        self.op_history.append(op_record)
        
        # Simulate computation time
        time.sleep(max(estimated_time, self.latency))
        
        # Create output tensor
        result = Tensor(
            shape=output_shape,
            name=f"matmul_{len(self.op_history)}",
            gpu=self
        )
        
        logger.debug(f"MatMul: {a.shape} @ {b.shape} -> {result.shape} (est. {estimated_time*1000:.2f}ms)")
        return result
    
    def elementwise_op(self, a, op_name: str) -> 'Tensor':
        """Simulate elementwise operations (ReLU, Sigmoid, etc.)"""
        from tensor_ops import Tensor  # Import here to avoid circular imports
        
        # Estimate FLOPS - typically 1 operation per element
        total_elements = np.prod(a.shape)
        total_flops = total_elements
        
        # Estimate time
        estimated_time = (total_flops / self.flops) * (1.05 + 0.05 * np.random.random())
        
        # Record operation
        op_record = {
            "op": op_name,
            "input_shape": a.shape,
            "output_shape": a.shape,
            "flops": total_flops,
            "estimated_time": estimated_time
        }
        self.op_history.append(op_record)
        
        # Simulate computation time
        time.sleep(max(estimated_time, self.latency * 0.1))
        
        # Create output tensor
        result = Tensor(
            shape=a.shape,
            name=f"{op_name}_{len(self.op_history)}",
            gpu=self
        )
        
        logger.debug(f"{op_name}: {a.shape} -> {result.shape} (est. {estimated_time*1000:.2f}ms)")
        return result
    
    def attention(self, query, key, value, mask=None) -> 'Tensor':
        """Simulate attention computation"""
        from tensor_ops import Tensor  # Import here to avoid circular imports
        
        # Validate shapes
        batch_size, seq_len, d_model = query.shape
        
        # 1. QK^T computation
        qk_estimated_flops = batch_size * seq_len * seq_len * d_model * 2
        
        # 2. Softmax
        softmax_estimated_flops = batch_size * seq_len * seq_len * 10  # exp, sum, division
        
        # 3. Attention * Value
        av_estimated_flops = batch_size * seq_len * seq_len * d_model * 2
        
        total_flops = qk_estimated_flops + softmax_estimated_flops + av_estimated_flops
        estimated_time = (total_flops / self.flops) * (1.2 + 0.2 * np.random.random())
        
        # Record operation
        op_record = {
            "op": "attention",
            "input_shapes": [query.shape, key.shape, value.shape],
            "output_shape": (batch_size, seq_len, d_model),
            "mask": mask is not None,
            "flops": total_flops,
            "estimated_time": estimated_time
        }
        self.op_history.append(op_record)
        
        # Simulate computation time
        time.sleep(max(estimated_time, self.latency))
        
        # Create output tensor
        result = Tensor(
            shape=(batch_size, seq_len, d_model),
            name=f"attention_{len(self.op_history)}",
            gpu=self
        )
        
        logger.debug(f"Attention: {query.shape} -> {result.shape} (est. {estimated_time*1000:.2f}ms)")
        return result
    
    def reset_stats(self) -> None:
        """Reset operation history and statistics"""
        self.op_history = []
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from operation history"""
        if not self.op_history:
            return {"message": "No operations recorded yet"}
        
        total_flops = sum(op["flops"] for op in self.op_history)
        total_time = sum(op["estimated_time"] for op in self.op_history)
        
        op_counts = {}
        for op in self.op_history:
            op_type = op["op"]
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        return {
            "total_operations": len(self.op_history),
            "operation_counts": op_counts,
            "total_flops": total_flops,
            "total_time": total_time,
            "effective_tflops": total_flops / total_time / 10**12 if total_time > 0 else 0,
        }
    
    def __str__(self) -> str:
        return (f"GPUSimulator(device={self.device_id}, "
                f"memory={self.allocated_memory}/{self.memory_size}MB, "
                f"capability={self.compute_capability})")