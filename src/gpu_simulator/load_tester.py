"""
Load Tester - Tools for load testing the GPU simulation
"""
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from collections import defaultdict

from gpu_simulator import GPUSimulator
from multi_gpu import MultiGPUManager
from tensor_ops import Tensor, calculate_tensor_size

logger = logging.getLogger('gpu_simulator.load_tester')

# Model size configurations (parameter count to architecture mapping)
MODEL_CONFIGS = {
    # Small models
    "125M": {
        "num_layers": 12,
        "hidden_size": 768,
        "ffn_hidden_size": 3072,
        "num_attention_heads": 12,
        "vocab_size": 50257,
        "seq_length": 1024,
    },
    "1B": {
        "num_layers": 24,
        "hidden_size": 1536,
        "ffn_hidden_size": 6144,
        "num_attention_heads": 16,
        "vocab_size": 50257,
        "seq_length": 1024,
    },
    # Medium models
    "7B": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 11008,
        "num_attention_heads": 32,
        "vocab_size": 50257,
        "seq_length": 2048,
    },
    "13B": {
        "num_layers": 40,
        "hidden_size": 5120,
        "ffn_hidden_size": 13824,
        "num_attention_heads": 40,
        "vocab_size": 50257,
        "seq_length": 2048,
    },
    # Large models
    "70B": {
        "num_layers": 80,
        "hidden_size": 8192,
        "ffn_hidden_size": 22016,
        "num_attention_heads": 64,
        "vocab_size": 50257,
        "seq_length": 4096,
    },
}

class LoadTester:
    """Load testing utilities for the GPU simulator"""
    
    def __init__(self, num_gpus: int = 1, memory_per_gpu: int = 16 * 1024):
        """
        Initialize the load tester
        
        Args:
            num_gpus: Number of GPUs to simulate
            memory_per_gpu: Memory per GPU in MB
        """
        self.gpu_manager = MultiGPUManager(
            num_gpus=num_gpus,
            memory_per_gpu=memory_per_gpu
        )
        
        # Tracking test results
        self.test_results = {}
        
        logger.info(f"Initialized LoadTester with {num_gpus} GPUs ({memory_per_gpu}MB each)")
    
    def calculate_model_memory(self, model_size: str, 
                              precision: str = "float16",
                              batch_size: int = 1,
                              sequence_length: Optional[int] = None) -> Dict:
        """
        Calculate memory requirements for a given model configuration
        
        Args:
            model_size: Model size key (e.g., "7B", "13B")
            precision: Data type precision (float32, float16, int8)
            batch_size: Batch size
            sequence_length: Sequence length (uses default if None)
            
        Returns:
            Dict: Memory calculations
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Available options: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_size].copy()
        
        # Override sequence length if provided
        if sequence_length is not None:
            config["seq_length"] = sequence_length
        
        # Bytes per parameter based on precision
        bytes_per_param = 4  # float32
        if precision == "float16":
            bytes_per_param = 2
        elif precision == "int8":
            bytes_per_param = 1
        
        # Model parameters memory
        # 1. Embedding layer: vocab_size * hidden_size
        embedding_params = config["vocab_size"] * config["hidden_size"]
        
        # 2. Transformer layers
        # Each layer has:
        #   - 3 attention weight matrices: 3 * hidden_size * hidden_size
        #   - 1 attention output projection: hidden_size * hidden_size
        #   - 2 FFN matrices: hidden_size * ffn_hidden_size + ffn_hidden_size * hidden_size
        #   - Layer norms: 4 * hidden_size (small, often negligible)
        params_per_layer = (
            3 * config["hidden_size"] * config["hidden_size"] +  # attention
            config["hidden_size"] * config["hidden_size"] +      # attn output
            config["hidden_size"] * config["ffn_hidden_size"] +  # FFN up
            config["ffn_hidden_size"] * config["hidden_size"] +  # FFN down
            4 * config["hidden_size"]                            # layer norms
        )
        transformer_params = config["num_layers"] * params_per_layer
        
        # 3. Output embedding (often tied with input embedding)
        output_params = 0  # Assuming tied embeddings
        
        # Total parameters
        total_params = embedding_params + transformer_params + output_params
        
        # Memory calculations
        model_memory_bytes = total_params * bytes_per_param
        model_memory_mb = model_memory_bytes / (1024 * 1024)
        
        # Activation memory (rough estimate)
        # For each layer: sequence_length * batch_size * hidden_size * 4 (activations) * bytes_per_value
        activation_memory_per_layer = (
            config["seq_length"] * batch_size * config["hidden_size"] * 4 * bytes_per_param
        )
        activation_memory_bytes = config["num_layers"] * activation_memory_per_layer
        activation_memory_mb = activation_memory_bytes / (1024 * 1024)
        
        # KV cache for inference
        # 2 (K and V) * batch_size * num_layers * seq_length * (hidden_size / num_attention_heads) * num_attention_heads * bytes_per_param
        kv_cache_bytes = (
            2 * batch_size * config["num_layers"] * config["seq_length"] * 
            config["hidden_size"] * bytes_per_param
        )
        kv_cache_mb = kv_cache_bytes / (1024 * 1024)
        
        # Optimizer states (for training)
        # For Adam: 2 states per parameter (momentum and variance) plus master weights if mixed precision
        optimizer_states_bytes = total_params * (2 * bytes_per_param + (4 if precision != "float32" else 0))
        optimizer_states_mb = optimizer_states_bytes / (1024 * 1024)
        
        # Gradient checkpointing reduces activation memory at cost of recomputation
        checkpointed_activation_mb = activation_memory_mb * 0.2  # ~80% reduction
        
        return {
            "model_size_name": model_size,
            "precision": precision,
            "batch_size": batch_size,
            "sequence_length": config["seq_length"],
            "total_parameters": total_params,
            "model_memory_mb": model_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "checkpointed_activation_mb": checkpointed_activation_mb,
            "kv_cache_mb": kv_cache_mb,
            "optimizer_states_mb": optimizer_states_mb,
            "training_memory_mb": model_memory_mb + activation_memory_mb + optimizer_states_mb,
            "training_checkpointed_mb": model_memory_mb + checkpointed_activation_mb + optimizer_states_mb,
            "inference_memory_mb": model_memory_mb + kv_cache_mb
        }
    
    def run_inference_test(self, 
                          batch_sizes: List[int],
                          model_size: str = "7B",
                          precision: str = "float16",
                          sequence_length: int = 2048,
                          num_iterations: int = 3) -> Dict:
        """
        Run inference load test with various batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            model_size: Model size key (e.g., "7B", "13B")
            precision: Data type precision (float32, float16, int8)
            sequence_length: Sequence length
            num_iterations: Number of iterations for each test
            
        Returns:
            Dict: Test results
        """
        results = {
            "test_type": "inference",
            "model_size": model_size,
            "precision": precision,
            "sequence_length": sequence_length,
            "num_gpus": self.gpu_manager.num_gpus,
            "batch_sizes": batch_sizes,
            "memory_usage": [],
            "throughput": [],
            "latency": [],
            "success": []
        }
        
        for batch_size in batch_sizes:
            # Reset GPU stats
            self.gpu_manager.reset_all()
            
            # Calculate memory requirements
            mem_req = self.calculate_model_memory(
                model_size=model_size,
                precision=precision,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            
            # Check if we have enough memory
            total_gpu_memory = self.gpu_manager.num_gpus * self.gpu_manager.memory_per_gpu
            if mem_req["inference_memory_mb"] > total_gpu_memory:
                logger.warning(f"Batch size {batch_size} would require {mem_req['inference_memory_mb']}MB, "
                              f"but only {total_gpu_memory}MB available across {self.gpu_manager.num_gpus} GPUs")
                results["memory_usage"].append(None)
                results["throughput"].append(None)
                results["latency"].append(None)
                results["success"].append(False)
                continue
            
            try:
                # Simulate loading the model
                logger.info(f"Testing batch size {batch_size} with {model_size} model...")
                
                # Create mock tensors for testing
                config = MODEL_CONFIGS[model_size]
                
                # Time the "inference"
                total_tokens = batch_size * sequence_length
                times = []
                
                for iteration in range(num_iterations):
                    # Simulate encoder input
                    input_shape = (batch_size, sequence_length, config["hidden_size"])
                    input_tensor = Tensor(shape=input_shape, name="input_ids", gpu=self.gpu_manager.gpus[0])
                    
                    # Timestamp before
                    start_time = time.time()
                    
                    # Simulated inference through layers
                    hidden_states = input_tensor
                    for layer_idx in range(min(3, config["num_layers"])):  # Simulate only a few layers
                        # Self-attention
                        query = Tensor(shape=input_shape, name=f"query_{layer_idx}", gpu=self.gpu_manager.gpus[0])
                        key = Tensor(shape=input_shape, name=f"key_{layer_idx}", gpu=self.gpu_manager.gpus[0])
                        value = Tensor(shape=input_shape, name=f"value_{layer_idx}", gpu=self.gpu_manager.gpus[0])
                        
                        attn_output = self.gpu_manager.gpus[0].attention(query, key, value)
                        
                        # FFN
                        ffn_input = Tensor(shape=input_shape, name=f"ffn_input_{layer_idx}", gpu=self.gpu_manager.gpus[0])
                        ffn_intermediate = Tensor(
                            shape=(batch_size, sequence_length, config["ffn_hidden_size"]),
                            name=f"ffn_intermediate_{layer_idx}",
                            gpu=self.gpu_manager.gpus[0]
                        )
                        ffn_output = Tensor(shape=input_shape, name=f"ffn_output_{layer_idx}", gpu=self.gpu_manager.gpus[0])
                        
                        # Update hidden states
                        hidden_states = ffn_output
                    
                    # Generate output logits for the last token
                    output_tensor = Tensor(
                        shape=(batch_size, config["vocab_size"]),
                        name="output_logits",
                        gpu=self.gpu_manager.gpus[0]
                    )
                    
                    # Timestamp after
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                latency = np.mean(times)
                throughput = total_tokens / latency
                
                # Get memory usage
                memory_stats = self.gpu_manager.get_memory_stats()["aggregate"]
                memory_usage = memory_stats["allocated_memory"]
                
                # Record results
                results["memory_usage"].append(memory_usage)
                results["throughput"].append(throughput)
                results["latency"].append(latency)
                results["success"].append(True)
                
                logger.info(f"Batch size {batch_size}: {throughput:.2f} tokens/sec, "
                           f"latency: {latency*1000:.2f}ms, memory: {memory_usage}MB")
                
            except Exception as e:
                logger.error(f"Error testing batch size {batch_size}: {str(e)}")
                results["memory_usage"].append(None)
                results["throughput"].append(None)
                results["latency"].append(None)
                results["success"].append(False)
        
        # Store results
        test_id = f"inference_{model_size}_{precision}_{sequence_length}_{int(time.time())}"
        self.test_results[test_id] = results
        
        return results
    
    def run_training_test(self,
                         batch_sizes: List[int],
                         model_size: str = "7B",
                         precision: str = "float16",
                         sequence_length: int = 2048,
                         use_grad_checkpointing: bool = True,
                         num_iterations: int = 3) -> Dict:
        """
        Run training load test with various batch sizes
        
        Args:
            batch_sizes: List of batch sizes to test
            model_size: Model size key (e.g., "7B", "13B")
            precision: Data type precision (float32, float16, int8)
            sequence_length: Sequence length
            use_grad_checkpointing: Whether to use gradient checkpointing
            num_iterations: Number of iterations for each test
            
        Returns:
            Dict: Test results
        """
        results = {
            "test_type": "training",
            "model_size": model_size,
            "precision": precision,
            "sequence_length": sequence_length,
            "grad_checkpointing": use_grad_checkpointing,
            "num_gpus": self.gpu_manager.num_gpus,
            "batch_sizes": batch_sizes,
            "memory_usage": [],
            "throughput": [],
            "iteration_time": [],
            "success": []
        }
        
        for batch_size in batch_sizes:
            # Reset GPU stats
            self.gpu_manager.reset_all()
            
            # Calculate memory requirements
            mem_req = self.calculate_model_memory(
                model_size=model_size,
                precision=precision,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            
            # Check if we have enough memory
            total_gpu_memory = self.gpu_manager.num_gpus * self.gpu_manager.memory_per_gpu
            memory_needed = mem_req["training_checkpointed_mb"] if use_grad_checkpointing else mem_req["training_memory_mb"]
            
            # Apply data parallel scaling factor (rough estimate)
            if self.gpu_manager.num_gpus > 1:
                memory_needed = memory_needed / self.gpu_manager.num_gpus
                memory_needed += mem_req["model_memory_mb"]  # Each GPU needs full model
            
            if memory_needed > total_gpu_memory:
                logger.warning(f"Batch size {batch_size} would require {memory_needed}MB, "
                              f"but only {total_gpu_memory}MB available across {self.gpu_manager.num_gpus} GPUs")
                results["memory_usage"].append(None)
                results["throughput"].append(None)
                results["iteration_time"].append(None)
                results["success"].append(False)
                continue
            
            try:
                # Simulate loading the model in data parallel mode if multiple GPUs
                if self.gpu_manager.num_gpus > 1:
                    self.gpu_manager.setup_data_parallel()
                
                logger.info(f"Testing batch size {batch_size} with {model_size} model (training)...")
                
                # Create mock tensors for testing
                config = MODEL_CONFIGS[model_size]
                
                # Time the "training" iterations
                total_tokens = batch_size * sequence_length
                times = []
                
                for iteration in range(num_iterations):
                    # Simulate encoder input and labels
                    input_shape = (batch_size, sequence_length, config["hidden_size"])
                    
                    # Distribute batches across GPUs
                    per_gpu_batch = max(1, batch_size // self.gpu_manager.num_gpus)
                    
                    # Track tensors per GPU
                    gpu_tensors = {}
                    for gpu_id in range(self.gpu_manager.num_gpus):
                        gpu = self.gpu_manager.gpus[gpu_id]
                        
                        # Create tensors on this GPU
                        per_gpu_shape = (per_gpu_batch, sequence_length, config["hidden_size"])
                        input_tensor = Tensor(shape=per_gpu_shape, name=f"input_ids_gpu{gpu_id}", gpu=gpu)
                        
                        # Store tensors for this GPU
                        gpu_tensors[gpu_id] = {
                            "input": input_tensor,
                            "hidden_states": input_tensor,  # Start with input as hidden states
                        }
                    
                    # Timestamp before
                    start_time = time.time()
                    
                    # Forward pass (simplified)
                    for layer_idx in range(min(3, config["num_layers"])):  # Simulate only a few layers
                        for gpu_id, tensors in gpu_tensors.items():
                            gpu = self.gpu_manager.gpus[gpu_id]
                            
                            # Get current hidden states
                            hidden_states = tensors["hidden_states"]
                            
                            # Self-attention
                            query = Tensor(shape=hidden_states.shape, name=f"query_{layer_idx}_gpu{gpu_id}", gpu=gpu)
                            key = Tensor(shape=hidden_states.shape, name=f"key_{layer_idx}_gpu{gpu_id}", gpu=gpu)
                            value = Tensor(shape=hidden_states.shape, name=f"value_{layer_idx}_gpu{gpu_id}", gpu=gpu)
                            
                            attn_output = gpu.attention(query, key, value)
                            
                            # FFN
                            ffn_intermediate_shape = (
                                hidden_states.shape[0],
                                hidden_states.shape[1],
                                config["ffn_hidden_size"]
                            )
                            ffn_intermediate = Tensor(
                                shape=ffn_intermediate_shape,
                                name=f"ffn_intermediate_{layer_idx}_gpu{gpu_id}",
                                gpu=gpu
                            )
                            ffn_output = Tensor(
                                shape=hidden_states.shape,
                                name=f"ffn_output_{layer_idx}_gpu{gpu_id}",
                                gpu=gpu
                            )
                            
                            # Update hidden states
                            tensors["hidden_states"] = ffn_output
                    
                    # Generate output logits
                    for gpu_id, tensors in gpu_tensors.items():
                        gpu = self.gpu_manager.gpus[gpu_id]
                        hidden_states = tensors["hidden_states"]
                        
                        # Final layer norm and output projection
                        output_tensor = Tensor(
                            shape=(hidden_states.shape[0], hidden_states.shape[1], config["vocab_size"]),
                            name=f"output_logits_gpu{gpu_id}",
                            gpu=gpu
                        )
                        
                        # Calculate loss
                        loss = Tensor(
                            shape=(1,),
                            name=f"loss_gpu{gpu_id}",
                            gpu=gpu
                        )
                        
                        # Store output
                        tensors["output"] = output_tensor
                        tensors["loss"] = loss
                    
                    # All-reduce the losses if data parallel
                    if self.gpu_manager.num_gpus > 1:
                        losses = {gpu_id: tensors["loss"] for gpu_id, tensors in gpu_tensors.items()}
                        first_gpu_loss = next(iter(losses.values()))
                        global_loss = self.gpu_manager.all_reduce(first_gpu_loss)
                    
                    # Backward pass (simplified)
                    for layer_idx in range(min(3, config["num_layers"])-1, -1, -1):
                        for gpu_id, tensors in gpu_tensors.items():
                            gpu = self.gpu_manager.gpus[gpu_id]
                            
                            # Calculate gradients for each tensor (simplified)
                            grad_shape = (
                                per_gpu_batch, 
                                sequence_length, 
                                config["hidden_size"]
                            )
                            grad_tensor = Tensor(
                                shape=grad_shape,
                                name=f"grad_{layer_idx}_gpu{gpu_id}",
                                gpu=gpu
                            )
                    
                    # All-reduce the gradients if data parallel
                    if self.gpu_manager.num_gpus > 1:
                        # Simulate gradient synchronization
                        for gpu_id, tensors in gpu_tensors.items():
                            gpu = self.gpu_manager.gpus[gpu_id]
                            grad_tensor = Tensor(
                                shape=(1, config["hidden_size"]),
                                name=f"final_grad_gpu{gpu_id}",
                                gpu=gpu
                            )
                            global_grad = self.gpu_manager.all_reduce(grad_tensor)
                    
                    # Optimizer step (simplified)
                    for gpu_id in gpu_tensors:
                        gpu = self.gpu_manager.gpus[gpu_id]
                        
                        # Optimizer states
                        opt_state1 = Tensor(
                            shape=(1, config["hidden_size"]),
                            name=f"opt_momentum_gpu{gpu_id}",
                            gpu=gpu
                        )
                        opt_state2 = Tensor(
                            shape=(1, config["hidden_size"]),
                            name=f"opt_variance_gpu{gpu_id}",
                            gpu=gpu
                        )
                    
                    # Timestamp after
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                iter_time = np.mean(times)
                throughput = total_tokens / iter_time
                
                # Get memory usage
                memory_stats = self.gpu_manager.get_memory_stats()["aggregate"]
                memory_usage = memory_stats["allocated_memory"]
                
                # Record results
                results["memory_usage"].append(memory_usage)
                results["throughput"].append(throughput)
                results["iteration_time"].append(iter_time)
                results["success"].append(True)
                
                logger.info(f"Batch size {batch_size} (training): {throughput:.2f} tokens/sec, "
                           f"iteration time: {iter_time*1000:.2f}ms, memory: {memory_usage}MB")
                
            except Exception as e:
                logger.error(f"Error testing batch size {batch_size} (training): {str(e)}")
                results["memory_usage"].append(None)
                results["throughput"].append(None)
                results["iteration_time"].append(None)
                results["success"].append(False)
        
        # Store results
        test_id = f"training_{model_size}_{precision}_{sequence_length}_{int(time.time())}"
        self.test_results[test_id] = results
        
        return results
    
    def run_generation_benchmark(self,
                               model_size: str = "7B",
                               precisions: List[str] = ["float16"],
                               sequence_lengths: List[int] = [1024, 2048, 4096],
                               batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """
        Run a comprehensive generation benchmark across different configurations
        
        Args:
            model_size: Model size key (e.g., "7B", "13B")
            precisions: List of precision formats to test
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dict: Combined benchmark results
        """
        benchmark_results = {
            "model_size": model_size,
            "precisions": precisions,
            "sequence_lengths": sequence_lengths,
            "batch_sizes": batch_sizes,
            "results": {}
        }
        
        for precision in precisions:
            for seq_len in sequence_lengths:
                # Run inference test
                result = self.run_inference_test(
                    batch_sizes=batch_sizes,
                    model_size=model_size,
                    precision=precision,
                    sequence_length=seq_len
                )
                
                # Store in combined results
                result_key = f"{precision}_{seq_len}"
                benchmark_results["results"][result_key] = result
        
        return benchmark_results
    
    def plot_results(self, results: Dict, plot_type: str = "throughput") -> None:
        """
        Plot the test results
        
        Args:
            results: Test results dictionary
            plot_type: Type of plot (throughput, memory, latency)
        """
        if not results or "batch_sizes" not in results:
            logger.error("Invalid results format for plotting")
            return
        
        batch_sizes = []
        y_values = []
        
        # Filter out failed tests
        for i, success in enumerate(results.get("success", [])):
            if success and i < len(results["batch_sizes"]):
                batch_sizes.append(results["batch_sizes"][i])
                
                if plot_type == "throughput":
                    y_values.append(results["throughput"][i])
                    y_label = "Throughput (tokens/sec)"
                    title = f"{results.get('test_type', 'Test')} Throughput - {results.get('model_size', 'Model')}"
                elif plot_type == "memory":
                    y_values.append(results["memory_usage"][i])
                    y_label = "Memory Usage (MB)"
                    title = f"{results.get('test_type', 'Test')} Memory Usage - {results.get('model_size', 'Model')}"
                elif plot_type == "latency":
                    if "latency" in results:
                        y_values.append(results["latency"][i] * 1000)  # Convert to ms
                        y_label = "Latency (ms)"
                    else:
                        y_values.append(results["iteration_time"][i] * 1000)  # Convert to ms
                        y_label = "Iteration Time (ms)"
                    title = f"{results.get('test_type', 'Test')} Latency - {results.get('model_size', 'Model')}"
        
        if not batch_sizes or not y_values:
            logger.warning("No valid data points to plot")
            return
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, y_values, 'o-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel(y_label)
        plt.title(title)
        
        # Add details
        plt.figtext(0.02, 0.02, 
                   f"Model: {results.get('model_size', 'N/A')}, "
                   f"Precision: {results.get('precision', 'N/A')}, "
                   f"Seq Length: {results.get('sequence_length', 'N/A')}, "
                   f"GPUs: {results.get('num_gpus', 1)}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, results_list: List[Dict], plot_type: str = "throughput") -> None:
        """
        Plot a comparison of multiple test results
        
        Args:
            results_list: List of test result dictionaries
            plot_type: Type of plot (throughput, memory, latency)
        """
        if not results_list:
            logger.error("Empty results list for comparison")
            return
        
        plt.figure(figsize=(12, 8))
        
        for results in results_list:
            if not results or "batch_sizes" not in results:
                continue
                
            batch_sizes = []
            y_values = []
            
            # Filter out failed tests
            for i, success in enumerate(results.get("success", [])):
                if success and i < len(results["batch_sizes"]):
                    batch_sizes.append(results["batch_sizes"][i])
                    
                    if plot_type == "throughput":
                        y_values.append(results["throughput"][i])
                        y_label = "Throughput (tokens/sec)"
                        title = "Throughput Comparison"
                    elif plot_type == "memory":
                        y_values.append(results["memory_usage"][i])
                        y_label = "Memory Usage (MB)"
                        title = "Memory Usage Comparison"
                    elif plot_type == "latency":
                        if "latency" in results:
                            y_values.append(results["latency"][i] * 1000)  # Convert to ms
                            y_label = "Latency (ms)"
                        else:
                            y_values.append(results["iteration_time"][i] * 1000)  # Convert to ms
                            y_label = "Iteration Time (ms)"
                        title = "Latency Comparison"
            
            if not batch_sizes or not y_values:
                continue
                
            # Create label for this result set
            label = f"{results.get('model_size', 'Model')}"
            if "precision" in results:
                label += f" ({results['precision']})"
            if "sequence_length" in results:
                label += f" seq={results['sequence_length']}"
            
            # Plot this result set
            plt.plot(batch_sizes, y_values, 'o-', linewidth=2, markersize=8, label=label)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results: Dict, filename: str = None) -> None:
        """
        Export test results to a file
        
        Args:
            results: Test results dictionary
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            # Generate a filename based on test parameters
            test_type = results.get("test_type", "test")
            model_size = results.get("model_size", "model")
            timestamp = int(time.time())
            filename = f"{test_type}_{model_size}_{timestamp}.json"
        
        import json
        
        try:
            # Convert numpy values to native Python types
            def convert_np(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(results, f, default=convert_np, indent=2)
                
            logger.info(f"Exported results to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
    
    def reset(self) -> None:
        """Reset the load tester state"""
        self.gpu_manager.reset_all()
        self.test_results = {}
        
        logger.info("Reset load tester state")