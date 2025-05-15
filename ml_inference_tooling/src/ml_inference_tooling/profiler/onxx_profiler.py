import os
import numpy as np
from typing import Dict, List, Any, Optional, Union
from .base_profiler import BaseProfiler

class ONNXProfiler(BaseProfiler):
    """
    Profiler for ONNX Runtime inference.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", batch_sizes: List[int] = [1, 2, 4, 8, 16], 
                 warmup_runs: int = 10, benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize ONNX profiler.
        
        Args:
            model_path: Path to the ONNX model file
            device: Device to run inference on ('cpu' or 'cuda')
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        super().__init__(model_path, device, batch_sizes, warmup_runs, benchmark_runs, config)
        
        # ONNX specific configurations
        self.providers = []
        if device == "cuda":
            self.providers = ['CUDAExecutionProvider']
        self.providers.append('CPUExecutionProvider')
        
        # Input shape and type information
        self.input_shape = config.get("input_shape", [1, 3, 224, 224])
        self.input_type = config.get("input_type", np.float32)
        self.input_name = config.get("input_name", None)  # Will be detected from model
        
    def load_model(self) -> Any:
        """
        Load the ONNX model.
        
        Returns:
            ONNX Runtime inference session
        """
        try:
            import onnxruntime as ort
            
            # Set session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            session = ort.InferenceSession(
                self.model_path, 
                sess_options=sess_options,
                providers=self.providers
            )
            
            # Get input name if not provided
            if not self.input_name:
                self.input_name = session.get_inputs()[0].name
                
            # Update input shape if not provided
            if self.input_shape[0] == 1:  # Only if batch size is default
                model_input = session.get_inputs()[0]
                self.input_shape = model_input.shape
                # Replace dynamic dimensions with fixed sizes
                self.input_shape = [dim if dim is not None else 1 for dim in self.input_shape]
                
            self.logger.info(f"Loaded ONNX model with input name: {self.input_name}, shape: {self.input_shape}")
            return session
            
        except ImportError:
            self.logger.error("onnxruntime not installed. Please install with 'pip install onnxruntime-gpu' for GPU support or 'pip install onnxruntime' for CPU only.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def prepare_input(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Prepare input data for ONNX inference.
        
        Args:
            batch_size: Batch size for the input
            
        Returns:
            Dictionary mapping input names to numpy arrays
        """
        # Update batch size in input shape
        current_shape = list(self.input_shape)
        current_shape[0] = batch_size
        
        # Create random input data
        input_data = np.random.rand(*current_shape).astype(self.input_type)
        
        return {self.input_name: input_data}
    
    def run_inference(self, input_data: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Run inference with ONNX Runtime.
        
        Args:
            input_data: Dictionary mapping input names to numpy arrays
            
        Returns:
            List of output arrays
        """
        outputs = self.model.run(None, input_data)
        return outputs