from typing import Dict, List, Any, Optional, Union
import numpy as np
from .base_profiler import BaseProfiler

class TorchScriptProfiler(BaseProfiler):
    """
    Profiler for TorchScript models.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", batch_sizes: List[int] = [1, 2, 4, 8, 16],
                 warmup_runs: int = 10, benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize TorchScript profiler.
        
        Args:
            model_path: Path to the TorchScript model file
            device: Device to run inference on ('cpu' or 'cuda')
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        super().__init__(model_path, device, batch_sizes, warmup_runs, benchmark_runs, config)
        
        # TorchScript specific configurations
        self.input_shape = config.get("input_shape", [1, 3, 224, 224])
        self.use_trace = config.get("use_trace", False)  # Whether to use torch.jit.trace for a model instead of loading
        self.model_class = config.get("model_class", None)  # Only used if use_trace=True
        
    def load_model(self) -> Any:
        """
        Load the TorchScript model.
        
        Returns:
            TorchScript module
        """
        try:
            import torch
            
            device = torch.device(self.device)
            
            if self.use_trace and self.model_class:
                # Load PyTorch model and trace it
                model = self.model_class()
                example_input = torch.rand(self.input_shape, device=device)
                traced_model = torch.jit.trace(model, example_input)
                traced_model = traced_model.to(device)
                return traced_model
            else:
                # Load traced/scripted model directly
                model = torch.jit.load(self.model_path)
                model = model.to(device)
                model.eval()  # Set to evaluation mode
                return model
                
        except ImportError:
            self.logger.error("PyTorch not installed. Please install with 'pip install torch'.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading TorchScript model: {str(e)}")
            raise
    
    def prepare_input(self, batch_size: int) -> Any:
        """
        Prepare input data for TorchScript inference.
        
        Args:
            batch_size: Batch size for the input
            
        Returns:
            PyTorch tensor
        """
        try:
            import torch
            
            # Update batch size in input shape
            current_shape = list(self.input_shape)
            current_shape[0] = batch_size
            
            # Create random input data
            input_data = torch.rand(current_shape, device=self.device)
            
            return input_data
            
        except ImportError:
            self.logger.error("PyTorch not installed. Please install with 'pip install torch'.")
            raise
    
    def run_inference(self, input_data: Any) -> Any:
        """
        Run inference with TorchScript.
        
        Args:
            input_data: PyTorch tensor
            
        Returns:
            PyTorch tensor output
        """
        with torch.no_grad():
            output = self.model(input_data)
        return output