"""
Multi-GPU - Simulation for multi-GPU environments
"""
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

from gpu_simulator import GPUSimulator

logger = logging.getLogger('gpu_simulator.multi_gpu')

class MultiGPUManager:
    """Manages multiple GPU simulators for distributed operations"""
    
    def __init__(self, num_gpus: int = 1, 
                 memory_per_gpu: int = 16 * 1024,  # 16GB
                 interconnect_bandwidth: int = 50):  # 50 GB/s NVLink-style
        """
        Initialize a multi-GPU environment
        
        Args:
            num_gpus: Number of GPUs to simulate
            memory_per_gpu: Memory per GPU in MB
            interconnect_bandwidth: GPU interconnect bandwidth in GB/s
        """
        self.num_gpus = max(1, num_gpus)
        self.memory_per_gpu = memory_per_gpu
        self.interconnect_bandwidth = interconnect_bandwidth
        
        # Create GPU simulators
        self.gpus: Dict[int, GPUSimulator] = {}
        for i in range(self.num_gpus):
            self.gpus[i] = GPUSimulator(
                device_id=i,
                memory_size=memory_per_gpu,
                bandwidth=interconnect_bandwidth,
            )
        
        # Track data parallelism state
        self.is_data_parallel = False
        self.dp_group_size = 1
        
        # Track model parallelism state
        self.is_model_parallel = False
        self.mp_group_size = 1
        
        logger.info(f"Initialized multi-GPU environment with {num_gpus} GPUs "
                    f"({memory_per_gpu}MB each, {interconnect_bandwidth}GB/s interconnect)")
    
    def get_gpu(self, device_id: int) -> GPUSimulator:
        """Get a specific GPU by device ID"""
        if device_id not in self.gpus:
            raise ValueError(f"Invalid GPU device ID: {device_id}")
        return self.gpus[device_id]
    
    def reset_all(self) -> None:
        """Reset all GPUs"""
        for gpu in self.gpus.values():
            gpu.reset_stats()
    
    def sync_all(self) -> None:
        """Synchronize all GPUs (simulate a barrier)"""
        # Simulate synchronization latency based on number of GPUs
        sync_time = 0.0001 * self.num_gpus * np.log2(self.num_gpus + 1)
        time.sleep(sync_time)
        logger.debug(f"Synchronized {self.num_gpus} GPUs")
    
    def all_reduce(self, tensor, op: str = "sum") -> 'Tensor':
        """
        Simulate all-reduce operation across GPUs
        
        Args:
            tensor: The tensor to reduce
            op: Reduction operation (sum, mean, max, min)
            
        Returns:
            Tensor: The reduced tensor on each GPU
        """
        from tensor_ops import Tensor
        
        # Validate operation
        if op not in ["sum", "mean", "max", "min"]:
            raise ValueError(f"Invalid reduction operation: {op}")
        
        # Get source GPU
        source_gpu = tensor.gpu
        if source_gpu is None:
            raise ValueError("Tensor must be on a GPU for all-reduce")
        
        # Calculate communication volume and time
        # All-reduce requires 2(n-1)/n data volume in optimized implementation
        data_size_gb = tensor.size_bytes / (1024 * 1024 * 1024)
        optimal_factor = 2 * (self.num_gpus - 1) / self.num_gpus
        comm_volume_gb = data_size_gb * optimal_factor
        
        # Calculate communication time based on bandwidth and latency
        comm_time = comm_volume_gb / self.interconnect_bandwidth
        comm_time += 0.0001 * np.log2(self.num_gpus)  # Additional latency factor
        
        # Simulate communication time
        time.sleep(comm_time)
        
        logger.debug(f"All-reduce: {tensor.shape}, op={op}, "
                     f"time={comm_time*1000:.2f}ms, {self.num_gpus} GPUs")
        
        # Create result tensor (same as input tensor for simulation purposes)
        result = Tensor(
            shape=tensor.shape,
            name=f"allreduce_{op}",
            dtype=tensor.dtype,
            gpu=source_gpu
        )
        
        return result
    
    def broadcast(self, tensor, src_gpu_id: int) -> Dict[int, 'Tensor']:
        """
        Simulate broadcasting a tensor from one GPU to all others
        
        Args:
            tensor: The tensor to broadcast
            src_gpu_id: Source GPU ID
            
        Returns:
            Dict[int, Tensor]: Map of GPU ID to tensor copy
        """
        from tensor_ops import Tensor
        
        # Validate source GPU
        if src_gpu_id not in self.gpus:
            raise ValueError(f"Invalid source GPU ID: {src_gpu_id}")
        
        # Get source GPU
        src_gpu = self.gpus[src_gpu_id]
        
        # Calculate communication volume and time
        data_size_gb = tensor.size_bytes / (1024 * 1024 * 1024)
        
        # Simulate using optimal tree-based broadcast algorithm
        # Time complexity is O(log(n)) where n is number of GPUs
        comm_time = (data_size_gb / self.interconnect_bandwidth) * np.log2(self.num_gpus)
        
        # Add latency factor
        comm_time += 0.0001 * np.log2(self.num_gpus)
        
        # Simulate communication time
        time.sleep(comm_time)
        
        logger.debug(f"Broadcast: {tensor.shape}, src={src_gpu_id}, "
                     f"time={comm_time*1000:.2f}ms, {self.num_gpus} GPUs")
        
        # Create result tensors on each GPU
        result = {}
        for gpu_id, gpu in self.gpus.items():
            if gpu_id == src_gpu_id:
                result[gpu_id] = tensor  # Original tensor on source GPU
            else:
                # Create a copy on target GPU
                result[gpu_id] = Tensor(
                    shape=tensor.shape,
                    name=f"bcast_from{src_gpu_id}",
                    dtype=tensor.dtype,
                    gpu=gpu
                )
        
        return result
    
    def scatter(self, tensors: Dict[int, 'Tensor'], src_gpu_id: int) -> None:
        """
        Simulate scattering different tensors from one GPU to others
        
        Args:
            tensors: Map of GPU ID to tensor to scatter to that GPU
            src_gpu_id: Source GPU ID
        """
        # Validate source GPU
        if src_gpu_id not in self.gpus:
            raise ValueError(f"Invalid source GPU ID: {src_gpu_id}")
        
        # Calculate total communication volume and time
        total_size_bytes = sum(t.size_bytes for t in tensors.values())
        data_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        
        # Communication time based on linear scatter
        comm_time = data_size_gb / self.interconnect_bandwidth
        
        # Add latency factor
        comm_time += 0.0001 * self.num_gpus
        
        # Simulate communication time
        time.sleep(comm_time)
        
        logger.debug(f"Scatter from GPU {src_gpu_id}, "
                     f"time={comm_time*1000:.2f}ms, {self.num_gpus} GPUs")
    
    def gather(self, tensors: Dict[int, 'Tensor'], dst_gpu_id: int) -> 'Tensor':
        """
        Simulate gathering tensors from all GPUs to one
        
        Args:
            tensors: Map of GPU ID to tensor to gather from that GPU
            dst_gpu_id: Destination GPU ID
            
        Returns:
            Tensor: The gathered tensor on the destination GPU
        """
        from tensor_ops import Tensor
        
        # Validate destination GPU
        if dst_gpu_id not in self.gpus:
            raise ValueError(f"Invalid destination GPU ID: {dst_gpu_id}")
        
        dst_gpu = self.gpus[dst_gpu_id]
        
        # Calculate total communication volume and time
        total_size_bytes = sum(t.size_bytes for t in tensors.values() if t.gpu.device_id != dst_gpu_id)
        data_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        
        # Communication time based on linear gather
        comm_time = data_size_gb / self.interconnect_bandwidth
        
        # Add latency factor
        comm_time += 0.0001 * self.num_gpus
        
        # Simulate communication time
        time.sleep(comm_time)
        
        logger.debug(f"Gather to GPU {dst_gpu_id}, "
                     f"time={comm_time*1000:.2f}ms, {len(tensors)} tensors")
        
        # For simulation, we'll use the first tensor's shape as the result shape
        if not tensors:
            raise ValueError("No tensors provided for gather operation")
        
        first_tensor = next(iter(tensors.values()))
        
        # Create result tensor on destination GPU
        result = Tensor(
            shape=first_tensor.shape,  # This would actually depend on the gather operation
            name="gather_result",
            dtype=first_tensor.dtype,
            gpu=dst_gpu
        )
        
        return result
    
    def setup_data_parallel(self, group_size: int = None) -> None:
        """
        Set up data parallelism across GPUs
        
        Args:
            group_size: Number of GPUs in each data parallel group
        """
        if group_size is None:
            group_size = self.num_gpus
        
        if group_size > self.num_gpus:
            raise ValueError(f"Data parallel group size {group_size} exceeds available GPUs {self.num_gpus}")
        
        self.is_data_parallel = True
        self.dp_group_size = group_size
        
        logger.info(f"Enabled data parallelism with group size {group_size}")
    
    def setup_model_parallel(self, group_size: int = None) -> None:
        """
        Set up model parallelism across GPUs
        
        Args:
            group_size: Number of GPUs for model parallelism
        """
        if group_size is None:
            group_size = self.num_gpus
        
        if group_size > self.num_gpus:
            raise ValueError(f"Model parallel group size {group_size} exceeds available GPUs {self.num_gpus}")
        
        self.is_model_parallel = True
        self.mp_group_size = group_size
        
        logger.info(f"Enabled model parallelism with group size {group_size}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics across all GPUs"""
        stats = {}
        for gpu_id, gpu in self.gpus.items():
            stats[gpu_id] = gpu.memory_usage()
        
        # Add aggregate stats
        total_memory = sum(s["total_memory"] for s in stats.values())
        allocated_memory = sum(s["allocated_memory"] for s in stats.values())
        
        stats["aggregate"] = {
            "total_memory": total_memory,
            "allocated_memory": allocated_memory,
            "free_memory": total_memory - allocated_memory,
            "utilization": (allocated_memory / total_memory) * 100 if total_memory > 0 else 0
        }
        
        return stats
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics across all GPUs"""
        stats = {}
        for gpu_id, gpu in self.gpus.items():
            stats[gpu_id] = gpu.get_performance_stats()
        
        # Calculate aggregate stats
        if all("total_flops" in s for s in stats.values()):
            total_flops = sum(s["total_flops"] for s in stats.values())
            total_time = max(s["total_time"] for s in stats.values() if "total_time" in s)
            
            stats["aggregate"] = {
                "total_flops": total_flops,
                "total_time": total_time,
                "effective_tflops": total_flops / total_time / 10**12 if total_time > 0 else 0,
            }
        
        return stats
    
    def __str__(self) -> str:
        mem_stats = self.get_memory_stats()["aggregate"]
        return (f"MultiGPUManager({self.num_gpus} GPUs, "
                f"{mem_stats['allocated_memory']}/{mem_stats['total_memory']}MB, "
                f"interconnect={self.interconnect_bandwidth}GB/s)")