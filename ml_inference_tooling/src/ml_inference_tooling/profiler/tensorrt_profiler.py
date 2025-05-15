from typing import Dict, List, Any, Optional, Union
import numpy as np
import os
from .base_profiler import BaseProfiler

class TensorRTProfiler(BaseProfiler):
    """
    Profiler for TensorRT inference.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", batch_sizes: List[int] = [1, 2, 4, 8, 16],
                 warmup_runs: int = 10, benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize TensorRT profiler.
        
        Args:
            model_path: Path to the TensorRT engine file or ONNX model to convert
            device: Device to run inference on (must be 'cuda' for TensorRT)
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        if device != "cuda":
            raise ValueError("TensorRT requires CUDA. CPU inference is not supported.")
            
        super().__init__(model_path, device, batch_sizes, warmup_runs, benchmark_runs, config)
        
        # TensorRT specific configurations
        self.input_shape = config.get("input_shape", [1, 3, 224, 224])
        self.input_name = config.get("input_name", "input")
        self.output_name = config.get("output_name", "output")
        self.precision = config.get("precision", "fp16")  # 'fp32', 'fp16', or 'int8'
        self.workspace_size = config.get("workspace_size", 1 << 30)  # 1GB
        self.is_engine = model_path.endswith('.engine') or model_path.endswith('.plan')
        self.engine_path = model_path if self.is_engine else model_path.replace('.onnx', '.engine')
        
    def load_model(self) -> Any:
        """
        Load or build TensorRT engine.
        
        Returns:
            TensorRT Engine and execution context
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Create CUDA context
            cuda_ctx = pycuda.autoinit.context
            
            # Initialize TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create runtime
            runtime = trt.Runtime(TRT_LOGGER)
            
            # Load or build engine
            if self.is_engine or os.path.exists(self.engine_path):
                # Load pre-built engine
                engine_path = self.model_path if self.is_engine else self.engine_path
                with open(engine_path, 'rb') as f:
                    engine_bytes = f.read()
                engine = runtime.deserialize_cuda_engine(engine_bytes)
                self.logger.info(f"Loaded TensorRT engine from {engine_path}")
            else:
                # Build engine from ONNX model
                self.logger.info(f"Building TensorRT engine from ONNX model: {self.model_path}")
                builder = trt.Builder(TRT_LOGGER)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, TRT_LOGGER)
                
                # Parse ONNX model
                with open(self.model_path, 'rb') as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            self.logger.error(f"ONNX parser error: {parser.get_error(error)}")
                        raise RuntimeError("Failed to parse ONNX model")
                
                # Configure builder
                config = builder.create_builder_config()
                config.max_workspace_size = self.workspace_size
                
                # Set precision
                if self.precision == "fp16" and builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    self.logger.info("Enabled FP16 precision")
                elif self.precision == "int8" and builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    # INT8 calibration would be needed here for production use
                    self.logger.info("Enabled INT8 precision")
                
                # Create optimization profiles for dynamic shapes if needed
                if network.num_inputs > 0 and any(dim is None for dim in network.get_input(0).shape):
                    profile = builder.create_optimization_profile()
                    input_tensor = network.get_input(0)
                    input_shape = input_tensor.shape
                    min_shape = input_shape.copy()
                    opt_shape = input_shape.copy()
                    max_shape = input_shape.copy()
                    
                    # Set batch size for each profile
                    if input_shape[0] is None:  # Dynamic batch size
                        min_shape[0] = 1
                        opt_shape[0] = 8  # Optimal batch size
                        max_shape[0] = max(self.batch_sizes)
                    
                    # Set any other dynamic dimensions
                    for i in range(1, len(input_shape)):
                        if input_shape[i] is None:
                            min_shape[i] = 1  # Minimum possible size
                            opt_shape[i] = self.input_shape[i]  # Use config value
                            max_shape[i] = self.input_shape[i] * 2  # Double for safety
                    
                    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                    config.add_optimization_profile(profile)
                
                # Build engine
                engine = builder.build_engine(network, config)
                
                # Save engine for future use
                with open(self.engine_path, 'wb') as f:
                    f.write(engine.serialize())
                self.logger.info(f"Saved TensorRT engine to {self.engine_path}")
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Get input and output binding information
            self.input_binding_idx = engine.get_binding_index(self.input_name) if self.input_name in [engine.get_binding_name(i) for i in range(engine.num_bindings)] else 0
            self.output_binding_idx = 1  # Assuming single output for simplicity
            
            # Verify input shape
            if not self.is_engine:  # Only for newly built engines
                binding_shape = engine.get_binding_shape(self.input_binding_idx)
                if binding_shape[0] == -1:  # Dynamic batch size
                    self.logger.info(f"Engine has dynamic batch size")
                    # Set default batch size in context
                    context.set_binding_shape(self.input_binding_idx, (self.batch_sizes[0], *binding_shape[1:]))
                else:
                    self.logger.info(f"Engine has fixed batch size: {binding_shape[0]}")
                
            return {"engine": engine, "context": context, "cuda_ctx": cuda_ctx}
            
        except ImportError:
            self.logger.error("TensorRT or PyCUDA not installed. Please install TensorRT from NVIDIA and PyCUDA.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading/building TensorRT engine: {str(e)}")
            raise
    
    def prepare_input(self, batch_size: int) -> Dict[str, Any]:
        """
        Prepare input data for TensorRT inference.
        
        Args:
            batch_size: Batch size for the input
            
        Returns:
            Dictionary with input/output bindings and buffers
        """
        try:
            import pycuda.driver as cuda
            
            engine = self.model["engine"]
            context = self.model["context"]
            
            # Set batch size for dynamic shape
            if engine.get_binding_shape(self.input_binding_idx)[0] == -1:
                new_shape = list(engine.get_binding_shape(self.input_binding_idx))
                new_shape[0] = batch_size
                context.set_binding_shape(self.input_binding_idx, new_shape)
            
            # Get shapes from context
            input_shape = context.get_binding_shape(self.input_binding_idx)
            output_shape = context.get_binding_shape(self.output_binding_idx)
            
            # Create host and device buffers
            h_input = np.random.rand(*input_shape).astype(np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            cuda.memcpy_htod(d_input, h_input)
            
            h_output = np.empty(output_shape, dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            # Create a list of device bindings
            bindings = [int(d_input), int(d_output)]
            
            return {
                "bindings": bindings,
                "h_input": h_input,
                "h_output": h_output,
                "d_input": d_input,
                "d_output": d_output,
                "batch_size": batch_size
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing TensorRT input: {str(e)}")
            raise
    
    def run_inference(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Run inference with TensorRT.
        
        Args:
            input_data: Dictionary with input/output bindings and buffers
            
        Returns:
            Numpy array output
        """
        try:
            import pycuda.driver as cuda
            
            context = self.model["context"]
            bindings = input_data["bindings"]
            batch_size = input_data["batch_size"]
            h_output = input_data["h_output"]
            d_output = input_data["d_output"]
            
            # Run inference
            context.execute_v2(bindings)
            
            # Copy output from device to host
            cuda.memcpy_dtoh(h_output, d_output)
            
            return h_output
            
        except Exception as e:
            self.logger.error(f"Error during TensorRT inference: {str(e)}")
            raise
    
    def cleanup(self):
        """
        Free CUDA memory and destroy context.
        """
        try:
            self.logger.info("Cleaning up TensorRT resources")
            del self.model["engine"]
            del self.model["context"]
            # Don't pop the CUDA context yet, as other profilers might still use it
            
        except Exception as e:
            self.logger.warning(f"Error during TensorRT cleanup: {str(e)}")