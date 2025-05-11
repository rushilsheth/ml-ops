import os
import argparse
import json
import sys
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.onnx_profiler import ONNXProfiler
from profiler.torchscript_profiler import TorchScriptProfiler
from profiler.triton_profiler import TritonProfiler
from profiler.vllm_profiler import vLLMProfiler
from profiler.tensorrt_profiler import TensorRTProfiler
from utils.metrics import MetricsAnalyzer
from utils.device_info import DeviceInfo
from config.models_config import get_full_config, get_benchmarking_models

def download_model(model_config: Dict[str, Any], output_dir: str) -> str:
    """
    Download a model if URL is provided.
    
    Args:
        model_config: Model configuration with URL
        output_dir: Directory to save model
        
    Returns:
        Path to the model file
    """
    import urllib.request
    
    # Return if no URL provided
    if "model_url" not in model_config or not model_config["model_url"]:
        return model_config.get("model_path", "")
    
    url = model_config["model_url"]
    filename = url.split("/")[-1]
    output_path = os.path.join(output_dir, filename)
    
    # Only download if file doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading model from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Model downloaded to {output_path}")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return ""
            
    return output_path

def profile_model(model_name: str, frameworks: List[str], output_dir: str, 
                 device: str = "cuda", batch_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Profile a model using multiple frameworks.
    
    Args:
        model_name: Name of the model to profile
        frameworks: List of frameworks to use
        output_dir: Directory to save results
        device: Device to use for inference
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary of profiling results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save system information
    DeviceInfo.print_summary()
    DeviceInfo.save_system_info(os.path.join(output_dir, "system_info.json"))
    
    results = {}
    
    # Profile with each framework
    for framework in frameworks:
        try:
            print(f"\n=== Profiling {model_name} with {framework} ===")
            
            # Get framework-specific configuration
            try:
                config = get_full_config(model_name, framework)
            except ValueError as e:
                print(f"Skipping {framework}: {str(e)}")
                continue
                
            # Override batch sizes if provided
            if batch_sizes:
                config["batch_sizes"] = batch_sizes
                
            # Create models directory
            models_dir = os.path.join(output_dir, "models")
            
            # Framework-specific profiling
            if framework == "onnx":
                # Download or get model path
                model_path = download_model(config, models_dir)
                if not model_path:
                    print(f"Skipping ONNX: model path not found")
                    continue
                    
                # Create profiler
                profiler = ONNXProfiler(
                    model_path=model_path, 
                    device=device,
                    batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
                    warmup_runs=config.get("warmup_runs", 10),
                    benchmark_runs=config.get("benchmark_runs", 50),
                    config={
                        "input_shape": config.get("input_shape"),
                        "input_name": config.get("input_name"),
                        "input_type": config.get("datatype", "FP32")
                    }
                )
                
            elif framework == "torchscript":
                # Download or get model path
                model_path = download_model(config, models_dir)
                if not model_path and not config.get("model_class"):
                    print(f"Skipping TorchScript: neither model path nor model class provided")
                    continue
                    
                # Create profiler
                profiler = TorchScriptProfiler(
                    model_path=model_path, 
                    device=device,
                    batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
                    warmup_runs=config.get("warmup_runs", 10),
                    benchmark_runs=config.get("benchmark_runs", 50),
                    config={
                        "input_shape": config.get("input_shape"),
                        "use_trace": config.get("use_trace", False),
                        "model_class": config.get("model_class")
                    }
                )
                
            elif framework == "triton":
                # Create profiler
                profiler = TritonProfiler(
                    model_name=config.get("model_name", model_name),
                    url=config.get("url", "localhost:8000"),
                    device=device,
                    batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
                    warmup_runs=config.get("warmup_runs", 10),
                    benchmark_runs=config.get("benchmark_runs", 50),
                    config={
                        "input_shape": config.get("input_shape"),
                        "input_name": config.get("input_name"),
                        "output_name": config.get("output_name"),
                        "protocol": config.get("protocol", "http"),
                        "datatype": config.get("datatype", "FP32")
                    }
                )
                
            elif framework == "vllm":
                if device != "cuda":
                    print(f"Skipping vLLM: requires CUDA")
                    continue
                
                # Create profiler
                profiler = vLLMProfiler(
                    model_path=config.get("model_name", model_name),
                    device="cuda",  # vLLM only supports CUDA
                    batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
                    warmup_runs=config.get("warmup_runs", 5),
                    benchmark_runs=config.get("benchmark_runs", 20),
                    config={
                        "max_tokens": config.get("max_tokens", 128),
                        "temperature": config.get("temperature", 0.0),
                        "top_p": config.get("top_p", 1.0),
                        "prompt_template": config.get("prompt_template", "Tell me about {topic}")
                    }
                )
                
            elif framework == "tensorrt":
                if device != "cuda":
                    print(f"Skipping TensorRT: requires CUDA")
                    continue
                
                # Download or get model path
                model_path = download_model(config, models_dir)
                if not model_path:
                    print(f"Skipping TensorRT: model path not found")
                    continue
                
                # Create profiler
                profiler = TensorRTProfiler(
                    model_path=model_path,
                    device="cuda",  # TensorRT only supports CUDA
                    batch_sizes=config.get("batch_sizes", [1, 2, 4, 8]),
                    warmup_runs=config.get("warmup_runs", 10),
                    benchmark_runs=config.get("benchmark_runs", 50),
                    config={
                        "input_shape": config.get("input_shape"),
                        "input_name": config.get("input_name"),
                        "output_name": config.get("output_name"),
                        "precision": config.get("precision", "fp16"),
                        "workspace_size": config.get("workspace_size", 1 << 30)
                    }
                )
                
            else:
                print(f"Skipping unknown framework: {framework}")
                continue
                
            # Run profiling
            result = profiler.run_profiling()
            
            # Save results
            result_path = os.path.join(output_dir, f"{model_name}_{framework}_{device}.json")
            profiler.save_results(result_path)
            
            # Visualize results
            profiler.plot_results(output_dir)
            
            # Store results
            results[framework] = result
            
        except Exception as e:
            import traceback
            print(f"Error profiling {model_name} with {framework}: {str(e)}")
            print(traceback.format_exc())
    
    # Compare results across frameworks
    if len(results) > 1:
        print("\n=== Comparing Results Across Frameworks ===")
        analyzer = MetricsAnalyzer()
        for result in results.values():
            analyzer.add_result(result)
            
        # Generate plots
        analyzer.plot_latency_comparison(os.path.join(output_dir, f"{model_name}_latency_comparison.png"))
        analyzer.plot_throughput_comparison(os.path.join(output_dir, f"{model_name}_throughput_comparison.png"))
        analyzer.plot_memory_comparison(os.path.join(output_dir, f"{model_name}_memory_comparison.png"))
        analyzer.plot_load_time_comparison(os.path.join(output_dir, f"{model_name}_load_time_comparison.png"))
        analyzer.plot_scaling_efficiency(os.path.join(output_dir, f"{model_name}_scaling_efficiency.png"))
        
        # Generate summary report
        analyzer.generate_summary_report(os.path.join(output_dir, f"{model_name}_summary_report.json"))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Profile ML models across different inference frameworks")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name to profile")
    parser.add_argument("--frameworks", type=str, nargs="+", 
                        default=["onnx", "torchscript", "tensorrt"],
                        help="Frameworks to profile")
    parser.add_argument("--output-dir", type=str, default="./profiling_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    
    args = parser.parse_args()
    
    # List available models and exit
    if args.list_models:
        models = get_benchmarking_models()
        print("\nAvailable models for benchmarking:")
        print("=================================")
        
        print("\nComputer Vision Models:")
        for model in models["cv"]:
            print(f"  - {model}")
            
        print("\nNLP Models:")
        for model in models["nlp"]:
            print(f"  - {model}")
            
        return
    
    # Profile the model
    profile_model(
        model_name=args.model,
        frameworks=args.frameworks,
        output_dir=args.output_dir,
        device=args.device,
        batch_sizes=args.batch_sizes
    )

if __name__ == "__main__":
    main()

# examples/profile_resnet.py
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.onnx_profiler import ONNXProfiler
from profiler.torchscript_profiler import TorchScriptProfiler
from profiler.tensorrt_profiler import TensorRTProfiler
from utils.model_converter import ModelConverter
from utils.metrics import MetricsAnalyzer
from utils.device_info import DeviceInfo

def main():
    parser = argparse.ArgumentParser(description="Profile ResNet model inference")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to ResNet model file (ONNX or PyTorch)")
    parser.add_argument("--output-dir", type=str, default="./profiling_results/resnet",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32],
                       help="Batch sizes to test")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save system information
    DeviceInfo.print_summary()
    DeviceInfo.save_system_info(os.path.join(args.output_dir, "system_info.json"))
    
    # Download or load model
    if args.model_path is None:
        import urllib.request
        import torch
        import torchvision.models as models
        
        # Download model weights
        model_path = os.path.join(args.output_dir, "resnet50_model")
        os.makedirs(model_path, exist_ok=True)
        
        # Load ResNet model from torchvision
        print("Loading ResNet50 model from torchvision...")
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # Save to TorchScript format
        torchscript_path = os.path.join(model_path, "resnet50.pt")
        if not os.path.exists(torchscript_path):
            print(f"Converting to TorchScript format...")
            example_input = torch.randn(1, 3, 224, 224)
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, torchscript_path)
            print(f"Saved TorchScript model to {torchscript_path}")
        
        # Convert to ONNX format
        onnx_path = os.path.join(model_path, "resnet50.onnx")
        if not os.path.exists(onnx_path):
            print(f"Converting to ONNX format...")
            onnx_path = ModelConverter.pytorch_to_onnx(
                model,
                input_shape=[1, 3, 224, 224],
                output_path=onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            print(f"Saved ONNX model to {onnx_path}")
    else:
        # Use provided model path
        if args.model_path.endswith(".onnx"):
            onnx_path = args.model_path
            # Convert to TorchScript if needed
            torchscript_path = args.model_path.replace(".onnx", ".pt")
        elif args.model_path.endswith(".pt") or args.model_path.endswith(".pth"):
            torchscript_path = args.model_path
            # Convert to ONNX if needed
            onnx_path = args.model_path.replace(".pt", ".onnx").replace(".pth", ".onnx")
            if not os.path.exists(onnx_path):
                print(f"Converting to ONNX format...")
                import torch
                model = torch.jit.load(torchscript_path)
                onnx_path = ModelConverter.pytorch_to_onnx(
                    model,
                    input_shape=[1, 3, 224, 224],
                    output_path=onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                )
        else:
            print(f"Unsupported model format: {args.model_path}")
            return
    
    # Profile with ONNX Runtime
    print("\n=== Profiling with ONNX Runtime ===")
    onnx_profiler = ONNXProfiler(
        model_path=onnx_path,
        device=args.device,
        batch_sizes=args.batch_sizes,
        warmup_runs=10,
        benchmark_runs=50,
        config={
            "input_shape": [1, 3, 224, 224],
            "input_name": "input"
        }
    )
    
    onnx_results = onnx_profiler.run_profiling()
    onnx_profiler.save_results(os.path.join(args.output_dir, f"resnet50_onnx_{args.device}.json"))
    onnx_profiler.plot_results(args.output_dir)
    
    # Profile with TorchScript
    print("\n=== Profiling with TorchScript ===")
    torchscript_profiler = TorchScriptProfiler(
        model_path=torchscript_path,
        device=args.device,
        batch_sizes=args.batch_sizes,
        warmup_runs=10,
        benchmark_runs=50,
        config={
            "input_shape": [1, 3, 224, 224]
        }
    )
    
    torchscript_results = torchscript_profiler.run_profiling()
    torchscript_profiler.save_results(os.path.join(args.output_dir, f"resnet50_torchscript_{args.device}.json"))
    torchscript_profiler.plot_results(args.output_dir)
    
    # Profile with TensorRT (if using CUDA)
    tensorrt_results = None
    if args.device == "cuda":
        print("\n=== Profiling with TensorRT ===")
        try:
            # Convert ONNX to TensorRT if needed
            tensorrt_path = os.path.join(os.path.dirname(onnx_path), "resnet50.engine")
            if not os.path.exists(tensorrt_path):
                print(f"Converting to TensorRT format...")
                tensorrt_path = ModelConverter.onnx_to_tensorrt(
                    onnx_path=onnx_path,
                    output_path=tensorrt_path,
                    precision="fp16",
                    max_batch_size=max(args.batch_sizes)
                )
                
            tensorrt_profiler = TensorRTProfiler(
                model_path=tensorrt_path,
                device="cuda",
                batch_sizes=args.batch_sizes,
                warmup_runs=10,
                benchmark_runs=50,
                config={
                    "input_shape": [1, 3, 224, 224],
                    "input_name": "input",
                    "output_name": "output",
                    "precision": "fp16"
                }
            )
            
            tensorrt_results = tensorrt_profiler.run_profiling()
            tensorrt_profiler.save_results(os.path.join(args.output_dir, "resnet50_tensorrt_cuda.json"))
            tensorrt_profiler.plot_results(args.output_dir)
            
        except Exception as e:
            import traceback
            print(f"Error during TensorRT profiling: {str(e)}")
            print(traceback.format_exc())
    
    # Compare results
    print("\n=== Comparing Results ===")
    analyzer = MetricsAnalyzer()
    analyzer.add_result(onnx_results)
    analyzer.add_result(torchscript_results)
    if tensorrt_results:
        analyzer.add_result(tensorrt_results)
        
    # Generate plots
    analyzer.plot_latency_comparison(os.path.join(args.output_dir, "resnet50_latency_comparison.png"))
    analyzer.plot_throughput_comparison(os.path.join(args.output_dir, "resnet50_throughput_comparison.png"))
    analyzer.plot_memory_comparison(os.path.join(args.output_dir, "resnet50_memory_comparison.png"))
    analyzer.plot_load_time_comparison(os.path.join(args.output_dir, "resnet50_load_time_comparison.png"))
    analyzer.plot_scaling_efficiency(os.path.join(args.output_dir, "resnet50_scaling_efficiency.png"))
    
    # Generate summary report
    analyzer.generate_summary_report(os.path.join(args.output_dir, "resnet50_summary_report.json"))
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()