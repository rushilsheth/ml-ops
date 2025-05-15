import os
import logging
from typing import Dict, Optional, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelConverter")

class ModelConverter:
    """
    Utility for converting models between different formats.
    """
    
    @staticmethod
    def pytorch_to_onnx(model, input_shape: List[int], output_path: str, 
                        input_names: List[str] = ["input"], output_names: List[str] = ["output"],
                        dynamic_axes: Optional[Dict] = None, opset_version: int = 13) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Input shape for tracing
            output_path: Path to save ONNX model
            input_names: Names of input tensors
            output_names: Names of output tensors
            dynamic_axes: Dynamic axes configuration
            opset_version: ONNX opset version
            
        Returns:
            Path to the saved ONNX model
        """
        try:
            import torch
            
            logger.info(f"Converting PyTorch model to ONNX: {output_path}")
            
            # Ensure model is in evaluation mode
            model.eval()
            
            # Create random input for tracing
            dummy_input = torch.randn(input_shape)
            
            # Set dynamic axes if not provided
            if dynamic_axes is None and input_shape[0] > 1:
                # Make batch dimension dynamic by default
                dynamic_axes = {
                    input_names[0]: {0: "batch_size"},
                    output_names[0]: {0: "batch_size"}
                }
            
            # Export model to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                export_params=True,
                do_constant_folding=True,
                verbose=False
            )
            
            # Verify the ONNX model
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"PyTorch model successfully converted to ONNX: {output_path}")
            return output_path
            
        except ImportError as e:
            missing_package = str(e).split("'")[1]
            logger.error(f"Required package not installed: {missing_package}")
            raise
            
        except Exception as e:
            logger.error(f"Error converting PyTorch model to ONNX: {str(e)}")
            raise
    
    @staticmethod
    def pytorch_to_torchscript(model, input_shape: List[int], output_path: str, 
                               use_trace: bool = True) -> str:
        """
        Convert PyTorch model to TorchScript format.
        
        Args:
            model: PyTorch model
            input_shape: Input shape for tracing
            output_path: Path to save TorchScript model
            use_trace: Whether to use tracing (True) or scripting (False)
            
        Returns:
            Path to the saved TorchScript model
        """
        try:
            import torch
            
            logger.info(f"Converting PyTorch model to TorchScript: {output_path}")
            
            # Ensure model is in evaluation mode
            model.eval()
            
            if use_trace:
                # Use tracing
                dummy_input = torch.randn(input_shape)
                traced_model = torch.jit.trace(model, dummy_input)
                torch.jit.save(traced_model, output_path)
                logger.info(f"PyTorch model successfully traced to TorchScript: {output_path}")
            else:
                # Use scripting
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, output_path)
                logger.info(f"PyTorch model successfully scripted to TorchScript: {output_path}")
                
            return output_path
            
        except ImportError:
            logger.error("PyTorch not installed. Please install with 'pip install torch'.")
            raise
            
        except Exception as e:
            logger.error(f"Error converting PyTorch model to TorchScript: {str(e)}")
            raise
    
    @staticmethod
    def onnx_to_tensorrt(onnx_path: str, output_path: str, 
                         precision: str = "fp16", workspace_size: int = 1 << 30,
                         max_batch_size: int = 16) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size
            
        Returns:
            Path to the saved TensorRT engine
        """
        try:
            import tensorrt as trt
            
            logger.info(f"Converting ONNX model to TensorRT: {output_path}")
            
            # Initialize TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            # Set precision
            if precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # INT8 calibration would be needed here for production use
                logger.info("Enabled INT8 precision (calibration required for accurate results)")
            
            # Create optimization profiles for dynamic shapes if needed
            if network.num_inputs > 0 and any(dim is None for dim in network.get_input(0).shape):
                profile = builder.create_optimization_profile()
                input_tensor = network.get_input(0)
                input_shape = input_tensor.shape
                input_name = input_tensor.name
                
                # Handle dynamic batch dimension
                if input_shape[0] == -1:  # Dynamic batch size
                    min_shape = (1,) + tuple(1 if dim == -1 else dim for dim in input_shape[1:])
                    opt_shape = (max(1, max_batch_size // 2),) + tuple(16 if dim == -1 else dim for dim in input_shape[1:])
                    max_shape = (max_batch_size,) + tuple(32 if dim == -1 else dim for dim in input_shape[1:])
                    
                    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                    config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            logger.info(f"ONNX model successfully converted to TensorRT: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("TensorRT not installed. Please install TensorRT from NVIDIA.")
            raise
            
        except Exception as e:
            logger.error(f"Error converting ONNX model to TensorRT: {str(e)}")
            raise
    
    @staticmethod
    def create_triton_model_repository(model_path: str, framework: str, model_name: str, 
                                       output_path: str, max_batch_size: int = 16,
                                       input_shape: List[int] = [1, 3, 224, 224],
                                       input_name: str = "input", output_name: str = "output",
                                       instance_count: int = 1, device: str = "gpu") -> str:
        """
        Create a Triton Inference Server model repository.
        
        Args:
            model_path: Path to the model file
            framework: Framework type ('pytorch', 'onnx', 'tensorrt')
            model_name: Name for the model in Triton
            output_path: Path to the model repository
            max_batch_size: Maximum batch size
            input_shape: Input shape
            input_name: Input tensor name
            output_name: Output tensor name
            instance_count: Number of model instances
            device: Device type ('gpu' or 'cpu')
            
        Returns:
            Path to the model repository
        """
        import json
        
        logger.info(f"Creating Triton model repository at: {output_path}")
        
        # Create model repository structure
        model_repo_path = os.path.join(output_path, model_name)
        version_path = os.path.join(model_repo_path, "1")  # Version 1
        os.makedirs(version_path, exist_ok=True)
        
        # Determine backend and create config.pbtxt
        if framework == "pytorch":
            backend = "pytorch"
            platform = "pytorch_libtorch"
        elif framework == "onnx":
            backend = "onnxruntime"
            platform = "onnxruntime_onnx"
        elif framework == "tensorrt":
            backend = "tensorrt"
            platform = "tensorrt_plan"
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Create config.pbtxt
        config_path = os.path.join(model_repo_path, "config.pbtxt")
        with open(config_path, 'w') as f:
            f.write(f'name: "{model_name}"\n')
            f.write(f'backend: "{backend}"\n')
            f.write(f'platform: "{platform}"\n')
            
            if max_batch_size > 0:
                f.write(f'max_batch_size: {max_batch_size}\n')
            
            # Input configuration
            f.write('input [\n')
            f.write('  {\n')
            f.write(f'    name: "{input_name}"\n')
            f.write(f'    data_type: TYPE_FP32\n')
            dims_str = ', '.join(str(dim) for dim in input_shape[1:])  # Skip batch dimension
            f.write(f'    dims: [{dims_str}]\n')
            f.write('  }\n')
            f.write(']\n')
            
            # Output configuration
            f.write('output [\n')
            f.write('  {\n')
            f.write(f'    name: "{output_name}"\n')
            f.write(f'    data_type: TYPE_FP32\n')
            f.write(f'    dims: [-1]\n')  # Dynamic output shape
            f.write('  }\n')
            f.write(']\n')
            
            # Instance group configuration
            f.write('instance_group [\n')
            f.write('  {\n')
            f.write(f'    count: {instance_count}\n')
            kind = "KIND_GPU" if device == "gpu" else "KIND_CPU"
            f.write(f'    kind: {kind}\n')
            f.write('  }\n')
            f.write(']\n')
            
            # Dynamic batching configuration
            f.write('dynamic_batching {\n')
            f.write('  preferred_batch_size: [1, 4, 8, 16]\n')
            f.write('  max_queue_delay_microseconds: 50000\n')
            f.write('}\n')
        
        # Copy model file to version directory
        import shutil
        model_filename = os.path.basename(model_path)
        dest_path = os.path.join(version_path, model_filename)
        shutil.copy(model_path, dest_path)
        
        logger.info(f"Triton model repository created at: {output_path}")
        logger.info(f"Model accessible as: {model_name}")
        
        return output_path