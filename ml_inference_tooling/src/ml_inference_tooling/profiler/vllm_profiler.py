from typing import Dict, List, Any, Optional, Union
import numpy as np
from .base_profiler import BaseProfiler

class vLLMProfiler(BaseProfiler):
    """
    Profiler for vLLM inference engine.
    """
    
    def __init__(self, model_path: str, device: str = "cuda", batch_sizes: List[int] = [1, 2, 4, 8, 16],
                 warmup_runs: int = 10, benchmark_runs: int = 100, config: Dict = None):
        """
        Initialize vLLM profiler.
        
        Args:
            model_path: Path or name of the model (HF model id)
            device: Device to run inference on (must be 'cuda' for vLLM)
            batch_sizes: List of batch sizes to test
            warmup_runs: Number of warmup inference runs
            benchmark_runs: Number of benchmark inference runs
            config: Additional configuration parameters
        """
        if device != "cuda":
            raise ValueError("vLLM requires CUDA. CPU inference is not supported.")
            
        super().__init__(model_path, device, batch_sizes, warmup_runs, benchmark_runs, config)
        
        # vLLM specific configurations
        self.max_tokens = config.get("max_tokens", 128)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.prompt_template = config.get("prompt_template", "Tell me about {topic}")
        self.topics = config.get("topics", ["artificial intelligence", "machine learning", "deep learning", 
                                            "natural language processing", "computer vision"])
    
    def load_model(self) -> Any:
        """
        Load the vLLM model.
        
        Returns:
            vLLM LLM instance
        """
        try:
            from vllm import LLM, SamplingParams
            
            # Set up sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            
            # Load model
            llm = LLM(model=self.model_path)
            
            return llm
            
        except ImportError:
            self.logger.error("vLLM not installed. Please install with 'pip install vllm'.")
            raise
            
        except Exception as e:
            self.logger.error(f"Error loading vLLM model: {str(e)}")
            raise
    
    def prepare_input(self, batch_size: int) -> List[str]:
        """
        Prepare prompts for vLLM inference.
        
        Args:
            batch_size: Batch size (number of prompts)
            
        Returns:
            List of prompt strings
        """
        import random
        
        # Generate prompts
        prompts = []
        for _ in range(batch_size):
            topic = random.choice(self.topics)
            prompt = self.prompt_template.format(topic=topic)
            prompts.append(prompt)
            
        return prompts
    
    def run_inference(self, input_data: List[str]) -> Any:
        """
        Run inference with vLLM.
        
        Args:
            input_data: List of prompt strings
            
        Returns:
            vLLM GenerateResponse
        """
        outputs = self.model.generate(
            prompts=input_data,
            sampling_params=self.sampling_params
        )
        return outputs