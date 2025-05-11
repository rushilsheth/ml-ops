import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.onnx_profiler import ONNXProfiler
from profiler.vllm_profiler import vLLMProfiler
from utils.metrics import MetricsAnalyzer
from utils.device_info import DeviceInfo

def main():
    parser = argparse.ArgumentParser(description="Profile BERT model inference")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                       help="HuggingFace model name for BERT")
    parser.add_argument("--output-dir", type=str, default="./profiling_results/bert",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (cuda or cpu)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--sequence-length", type=int, default=128,
                       help="Input sequence length")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save system information
    DeviceInfo.print_summary()
    DeviceInfo.save_system_info(os.path.join(args.output_dir, "system_info.json"))
    
    # ONNX configuration
    onnx_config = {
        "model_path": None,  # We'll create this
        "input_shape": [1, args.sequence_length],
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "output_name": "output",
        "input_type": "INT64"
    }
    
    # vLLM configuration
    vllm_config = {
        "model_name": args.model_name,
        "max_tokens": 64,
        "temperature": 0.0,
        "prompt_template": "{text}"
    }
    
    # Export BERT model to ONNX
    onnx_path = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}.onnx")
    if not os.path.exists(onnx_path):
        try:
            print(f"Exporting {args.model_name} to ONNX format...")
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            model = AutoModelForMaskedLM.from_pretrained(args.model_name)
            model.eval()
            
            # Create dummy inputs
            dummy_input_ids = torch.ones(1, args.sequence_length, dtype=torch.long)
            dummy_attention_mask = torch.ones(1, args.sequence_length, dtype=torch.long)
            dummy_token_type_ids = torch.zeros(1, args.sequence_length, dtype=torch.long)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
                onnx_path,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "token_type_ids": {0: "batch_size"},
                    "output": {0: "batch_size"}
                },
                opset_version=12
            )
            print(f"Exported ONNX model to {onnx_path}")
            
        except Exception as e:
            import traceback
            print(f"Error exporting to ONNX: {str(e)}")
            print(traceback.format_exc())
            onnx_path = None
    
    results = []
    
    # Profile with ONNX Runtime if ONNX model is available
    if onnx_path and os.path.exists(onnx_path):
        print("\n=== Profiling with ONNX Runtime ===")
        try:
            onnx_profiler = ONNXProfiler(
                model_path=onnx_path,
                device=args.device,
                batch_sizes=args.batch_sizes,
                warmup_runs=5,
                benchmark_runs=20,
                config={
                    "input_shape": [1, args.sequence_length],
                    "input_name": "input_ids",
                    "input_type": "INT64"
                }
            )
            
            onnx_results = onnx_profiler.run_profiling()
            onnx_profiler.save_results(os.path.join(args.output_dir, f"bert_onnx_{args.device}.json"))
            onnx_profiler.plot_results(args.output_dir)
            results.append(onnx_results)
            
        except Exception as e:
            import traceback
            print(f"Error during ONNX profiling: {str(e)}")
            print(traceback.format_exc())
    
    # Profile with vLLM if using CUDA
    if args.device == "cuda":
        print("\n=== Profiling with vLLM ===")
        try:
            vllm_profiler = vLLMProfiler(
                model_path=args.model_name,
                device="cuda",
                batch_sizes=args.batch_sizes,
                warmup_runs=2,
                benchmark_runs=10,
                config=vllm_config
            )
            
            vllm_results = vllm_profiler.run_profiling()
            vllm_profiler.save_results(os.path.join(args.output_dir, "bert_vllm_cuda.json"))
            vllm_profiler.plot_results(args.output_dir)
            results.append(vllm_results)
            
        except Exception as e:
            import traceback
            print(f"Error during vLLM profiling: {str(e)}")
            print(traceback.format_exc())
    
    # Compare results if we have more than one
    if len(results) > 1:
        print("\n=== Comparing Results ===")
        analyzer = MetricsAnalyzer()
        for result in results:
            analyzer.add_result(result)
            
        # Generate plots
        analyzer.plot_latency_comparison(os.path.join(args.output_dir, "bert_latency_comparison.png"))
        analyzer.plot_throughput_comparison(os.path.join(args.output_dir, "bert_throughput_comparison.png"))
        analyzer.plot_memory_comparison(os.path.join(args.output_dir, "bert_memory_comparison.png"))
        analyzer.plot_load_time_comparison(os.path.join(args.output_dir, "bert_load_time_comparison.png"))
        
        # Generate summary report
        analyzer.generate_summary_report(os.path.join(args.output_dir, "bert_summary_report.json"))
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()