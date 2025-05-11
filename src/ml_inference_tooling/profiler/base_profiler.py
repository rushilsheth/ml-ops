import time
import numpy as np
import psutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseProfiler(ABC):
    """
    Base class for all inference profilers.
    Implements common profiling logic and metrics collection.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", batch_sizes: List[int] = [1, 2, 4, 8, 16], 
                 warmup_runs: int = 10, benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize the base profiler.
        
        Args:
            model_path: Path to the model file
            device: Device to run inference on ('cpu' or 'cuda')
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        self.model_path = model_path
        self.device = device
        self.batch_sizes = batch_sizes
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.model = None
        
        # Check if device is available
        if device == "cuda" and not self._is_cuda_available():
            self.logger.warning("CUDA is not available, falling back to CPU")
            self.device = "cpu"
            
        self.logger.info(f"Initialized {self.__class__.__name__} with device={self.device}")
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @abstractmethod
    def load_model(self) -> Any:
        """
        Load the model for inference.
        
        Returns:
            Loaded model object
        """
        pass
    
    @abstractmethod
    def prepare_input(self, batch_size: int) -> Any:
        """
        Prepare input data for the model.
        
        Args:
            batch_size: Batch size for the input
            
        Returns:
            Prepared input data
        """
        pass
    
    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        """
        Run a single inference pass.
        
        Args:
            input_data: Input data for inference
            
        Returns:
            Model output
        """
        pass
    
    def measure_latency(self, batch_size: int) -> Dict[str, float]:
        """
        Measure inference latency for a specific batch size.
        
        Args:
            batch_size: Batch size to measure
            
        Returns:
            Dictionary of latency metrics
        """
        self.logger.info(f"Measuring latency for batch_size={batch_size}")
        
        # Prepare input
        input_data = self.prepare_input(batch_size)
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            _ = self.run_inference(input_data)
            
        # Benchmark runs
        latencies = []
        for _ in range(self.benchmark_runs):
            start_time = time.time()
            _ = self.run_inference(input_data)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
        # Calculate statistics
        latencies = np.array(latencies)
        metrics = {
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "std": float(np.std(latencies)),
            "throughput": float(batch_size * 1000 / np.mean(latencies))  # items/second
        }
        
        return metrics
    
    def measure_memory(self, batch_size: int) -> Dict[str, float]:
        """
        Measure memory usage during inference.
        
        Args:
            batch_size: Batch size to measure
            
        Returns:
            Dictionary of memory metrics
        """
        self.logger.info(f"Measuring memory for batch_size={batch_size}")
        
        # Get baseline memory usage
        baseline_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        # Prepare input
        input_data = self.prepare_input(batch_size)
        
        # Measure memory during inference
        _ = self.run_inference(input_data)
        peak_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        memory_metrics = {
            "baseline_mb": baseline_memory,
            "peak_mb": peak_memory,
            "used_mb": peak_memory - baseline_memory
        }
        
        if self.device == "cuda":
            try:
                import torch
                memory_metrics["gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_metrics["gpu_reserved_mb"] = torch.cuda.max_memory_reserved() / 1024 / 1024
                torch.cuda.reset_peak_memory_stats()
            except (ImportError, AttributeError):
                self.logger.warning("Could not measure GPU memory usage")
                
        return memory_metrics
    
    def run_profiling(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the full profiling suite.
        
        Returns:
            Dictionary of profiling results
        """
        self.logger.info(f"Starting profiling with {self.__class__.__name__}")
        
        try:
            # Load model
            load_start = time.time()
            self.model = self.load_model()
            load_time = time.time() - load_start
            
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Run profiling for each batch size
            batch_results = {}
            for batch_size in self.batch_sizes:
                batch_results[str(batch_size)] = {
                    "latency": self.measure_latency(batch_size),
                    "memory": self.measure_memory(batch_size)
                }
                
            # Collect all results
            self.results = {
                "framework": self.__class__.__name__.replace("Profiler", ""),
                "device": self.device,
                "model_path": self.model_path,
                "load_time_seconds": load_time,
                "batch_results": batch_results,
                "system_info": self._get_system_info()
            }
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during profiling: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            "platform": {
                "system": os.name,
                "python_version": os.sys.version
            }
        }
        
        if self.device == "cuda":
            try:
                import torch
                info["gpu"] = {
                    "name": torch.cuda.get_device_name(0),
                    "count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda
                }
            except (ImportError, AttributeError):
                info["gpu"] = {"error": "Could not get GPU info"}
                
        return info
    
    def save_results(self, output_path: str) -> None:
        """
        Save profiling results to a file.
        
        Args:
            output_path: Path to save results
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")
    
    def plot_results(self, output_dir: str = None) -> None:
        """
        Plot profiling results.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.results:
            self.logger.warning("No results to plot")
            return
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Extract data for plotting
        batch_sizes = [int(bs) for bs in self.results["batch_results"].keys()]
        latency_means = [self.results["batch_results"][str(bs)]["latency"]["mean"] for bs in batch_sizes]
        throughputs = [self.results["batch_results"][str(bs)]["latency"]["throughput"] for bs in batch_sizes]
        
        # Plot latency vs batch size
        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, latency_means, 'o-', label='Mean Latency (ms)')
        plt.title(f'Latency vs Batch Size - {self.results["framework"]} ({self.device})')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{self.results['framework']}_{self.device}_latency.png"))
        plt.show()
        
        # Plot throughput vs batch size
        plt.figure(figsize=(10, 5))
        plt.plot(batch_sizes, throughputs, 'o-', label='Throughput (items/sec)')
        plt.title(f'Throughput vs Batch Size - {self.results["framework"]} ({self.device})')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (items/sec)')
        plt.grid(True)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{self.results['framework']}_{self.device}_throughput.png"))
        plt.show()