from typing import Dict, List, Any, Optional, Union
import numpy as np
import time
from .base_profiler import BaseProfiler

class TritonProfiler(BaseProfiler):
    """
    Profiler for Triton Inference Server.
    """
    
    def __init__(self, model_name: str, url: str = "localhost:8000", device: str = "cuda", 
                 batch_sizes: List[int] = [1, 2, 4, 8, 16], warmup_runs: int = 10, 
                 benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize Triton profiler.
        
        Args:
            model_name: Name of the model in Triton server
            url: Triton server URL (host:port)
            device: Device for performance comparison
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        # For Triton, we use model_name instead of model_path
        super().__init__(model_name, device, batch_sizes, warmup_runs, benchmark_runs, config)
        
        self.url = url
        self.model_name = model_name
        self.input_shape = config.get("input_shape", [1, 3, 224, 224])
        self.input_name = config.get("input_name", "input")
        self.output_name = config.get("output_name", "output")
        self.protocol = config.get("protocol", "http")  # 'http' or 'grpc'
        self.datatype = config.get("datatype", "FP32")
        
    def load_model(self) -> Any:
        """
        Connect to Triton server and create client.
        
        Returns:
            Triton client
        """
        try:
            if self.protocol == "http":
                from tritonclient.http import InferenceServerClient
                client = InferenceServerClient(url=self.url)
            else:  # grpc
                from tritonclient.grpc import InferenceServerClient
                client = InferenceServerClient(url=self.url)
                
            # Check server health
            if not client.is_server_live():
                raise RuntimeError("Triton server is not live")
                
            # Check if model is loaded
            if not client.is_model_ready(self.model_name):
                raise RuntimeError(f"Model {self.model_name} is not ready on the server")
                
            # Get model metadata
            model_metadata = client.get_model_metadata(self.model_name)
            self.logger.info(f"Connected to Triton server, model {self.model_name} is ready")
            
            return client
            
        except ImportError:
            self.logger.error("Triton client not installed. Please install with 'pip install tritonclient[all]'.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error connecting to Triton server: {str(e)}")
            raise
    
    def prepare_input(self, batch_size: int) -> Any:
        """
        Prepare input data for Triton inference.
        
        Args:
            batch_size: Batch size for the input
            
        Returns:
            InferInput object
        """
        try:
            # Update batch size in input shape
            current_shape = list(self.input_shape)
            current_shape[0] = batch_size
            
            # Create random input data
            input_data = np.random.rand(*current_shape).astype(np.float32)
            
            # Create InferInput
            if self.protocol == "http":
                from tritonclient.http import InferInput
                infer_input = InferInput(self.input_name, current_shape, self.datatype)
                infer_input.set_data_from_numpy(input_data)
            else:  # grpc
                from tritonclient.grpc import InferInput
                infer_input = InferInput(self.input_name, current_shape, self.datatype)
                infer_input.set_data_from_numpy(input_data)
                
            return [infer_input]
            
        except ImportError:
            self.logger.error("Triton client not installed. Please install with 'pip install tritonclient[all]'.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error preparing input for Triton: {str(e)}")
            raise
    
    def run_inference(self, input_data: Any) -> Any:
        """
        Run inference with Triton.
        
        Args:
            input_data: List of InferInput objects
            
        Returns:
            InferResult object
        """
        try:
            if self.protocol == "http":
                from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
                output = InferRequestedOutput(self.output_name)
                response = self.model.infer(model_name=self.model_name, 
                                         inputs=input_data,
                                         outputs=[output])
            else:  # grpc
                from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
                output = InferRequestedOutput(self.output_name)
                response = self.model.infer(model_name=self.model_name, 
                                         inputs=input_data,
                                         outputs=[output])
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error during Triton inference: {str(e)}")
            raise