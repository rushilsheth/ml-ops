"""
Tensor Operations - Simulates tensor operations for the GPU simulator
"""
import logging
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any

logger = logging.getLogger('gpu_simulator.tensor_ops')

class Tensor:
    """Simulates a tensor in GPU memory"""
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 name: str = "",
                 dtype: str = "float32",
                 gpu = None,
                 init_method: str = "zeros"):
        """
        Initialize a tensor
        
        Args:
            shape: Tensor shape as tuple
            name: Tensor name for tracking
            dtype: Data type (float32, float16, int32, etc.)
            gpu: GPUSimulator instance
            init_method: Initialization method (zeros, ones, random)
        """
        self.shape = shape
        self.name = name
        self.dtype = dtype
        self.gpu = gpu
        
        # Calculate size in MB
        self.elem_size = 4  # bytes for float32
        if dtype == "float16":
            self.elem_size = 2
        elif dtype == "int8":
            self.elem_size = 1
        
        self.num_elements = np.prod(shape)
        self.size_bytes = self.num_elements * self.elem_size
        self.size_mb = self.size_bytes / (1024 * 1024)
        
        # Allocate on GPU if available
        self.gpu_addr = None
        if gpu is not None:
            self.gpu_addr = gpu.allocate(self.size_mb, name)
            
        # Track creation time
        self.created_at = time.time()
        
        logger.debug(f"Created tensor {name} with shape {shape}, size {self.size_mb:.2f}MB")
    
    def __del__(self):
        """Clean up GPU memory when tensor is deleted"""
        if hasattr(self, 'gpu') and self.gpu is not None and hasattr(self, 'gpu_addr') and self.gpu_addr is not None:
            try:
                self.gpu.free(self.gpu_addr)
                logger.debug(f"Freed tensor {self.name} from GPU memory")
            except:
                pass  # Ignore errors during cleanup
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with another tensor"""
        if self.gpu is None or other.gpu is None:
            raise ValueError("Both tensors must be on a GPU")
        
        if self.gpu.device_id != other.gpu.device_id:
            raise ValueError("Tensors must be on the same GPU for matmul")
        
        return self.gpu.matmul(self, other)
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Overload @ operator for matrix multiplication"""
        return self.matmul(other)
    
    def relu(self) -> 'Tensor':
        """Apply ReLU activation"""
        if self.gpu is None:
            raise ValueError("Tensor must be on a GPU")
        
        return self.gpu.elementwise_op(self, "relu")
    
    def sigmoid(self) -> 'Tensor':
        """Apply sigmoid activation"""
        if self.gpu is None:
            raise ValueError("Tensor must be on a GPU")
        
        return self.gpu.elementwise_op(self, "sigmoid")
    
    def softmax(self, dim: int = -1) -> 'Tensor':
        """Apply softmax along specified dimension"""
        if self.gpu is None:
            raise ValueError("Tensor must be on a GPU")
        
        return self.gpu.elementwise_op(self, f"softmax_dim{dim}")
    
    def layernorm(self) -> 'Tensor':
        """Apply layer normalization"""
        if self.gpu is None:
            raise ValueError("Tensor must be on a GPU")
        
        return self.gpu.elementwise_op(self, "layernorm")
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """Reshape tensor to new dimensions"""
        if np.prod(new_shape) != self.num_elements:
            raise ValueError(f"Cannot reshape tensor of size {self.num_elements} to shape {new_shape}")
        
        # For reshapes, we don't allocate new memory
        result = Tensor(
            shape=new_shape,
            name=f"{self.name}_reshaped",
            dtype=self.dtype,
            gpu=None  # Don't allocate new GPU memory
        )
        
        # Share the GPU address
        result.gpu = self.gpu
        result.gpu_addr = self.gpu_addr
        
        logger.debug(f"Reshaped tensor from {self.shape} to {new_shape}")
        return result
    
    def to_gpu(self, gpu) -> 'Tensor':
        """Move tensor to a specific GPU"""
        if self.gpu is gpu:
            return self  # Already on the right GPU
        
        # Create a new tensor on the target GPU
        result = Tensor(
            shape=self.shape,
            name=f"{self.name}_gpu{gpu.device_id}",
            dtype=self.dtype,
            gpu=gpu
        )
        
        # Simulate data transfer
        data_size_gb = self.size_bytes / (1024 * 1024 * 1024)
        transfer_time = data_size_gb / min(self.gpu.bandwidth, gpu.bandwidth) if self.gpu else data_size_gb / gpu.bandwidth
        
        # Add some realistic latency
        transfer_time += 0.001  # 1ms base latency
        
        # Simulate the transfer time
        time.sleep(transfer_time)
        
        logger.debug(f"Transferred tensor from {self.gpu.device_id if self.gpu else 'CPU'} to GPU {gpu.device_id}")
        return result
    
    def clone(self) -> 'Tensor':
        """Create a copy of this tensor"""
        result = Tensor(
            shape=self.shape,
            name=f"{self.name}_clone",
            dtype=self.dtype,
            gpu=self.gpu
        )
        
        logger.debug(f"Cloned tensor {self.name} with shape {self.shape}")
        return result
    
    def __str__(self) -> str:
        gpu_info = f"GPU:{self.gpu.device_id}" if self.gpu else "CPU"
        return f"Tensor({self.name}, shape={self.shape}, {self.size_mb:.2f}MB, {gpu_info})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_random_tensor(shape: Tuple[int, ...], name: str = "random", gpu = None) -> Tensor:
    """Helper function to create a random tensor"""
    tensor = Tensor(shape=shape, name=name, gpu=gpu)
    return tensor


def create_parameter_tensor(shape: Tuple[int, ...], name: str = "parameter", gpu = None) -> Tensor:
    """Create a parameter tensor with He initialization scaling"""
    tensor = Tensor(shape=shape, name=name, gpu=gpu)
    return tensor


def zeros(shape: Tuple[int, ...], name: str = "zeros", gpu = None) -> Tensor:
    """Create a tensor filled with zeros"""
    return Tensor(shape=shape, name=name, gpu=gpu, init_method="zeros")


def ones(shape: Tuple[int, ...], name: str = "ones", gpu = None) -> Tensor:
    """Create a tensor filled with ones"""
    return Tensor(shape=shape, name=name, gpu=gpu, init_method="ones")


def calculate_tensor_size(shape: Tuple[int, ...], dtype: str = "float32") -> float:
    """Calculate tensor size in MB"""
    elem_size = 4  # bytes for float32
    if dtype == "float16":
        elem_size = 2
    elif dtype == "int8":
        elem_size = 1
    
    num_elements = np.prod(shape)
    size_bytes = num_elements * elem_size
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb