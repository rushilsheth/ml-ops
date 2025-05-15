# setup.py
from setuptools import setup, find_packages

setup(
    name="ml_inference_profiler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "onnx": ["onnxruntime-gpu>=1.8.0; platform_system!='Darwin'", "onnxruntime>=1.8.0"],
        "torch": ["torch>=1.9.0"],
        "triton": ["tritonclient[all]>=2.11.0"],
        "vllm": ["vllm>=0.1.0"],
        "tensorrt": [],  # No PyPI package for TensorRT
        "transformers": ["transformers>=4.15.0"],
        "all": [
            "onnxruntime-gpu>=1.8.0; platform_system!='Darwin'", 
            "onnxruntime>=1.8.0",
            "torch>=1.9.0",
            "tritonclient[all]>=2.11.0",
            "transformers>=4.15.0"
        ]
    },
    author="ML Engineer",
    author_email="user@example.com",
    description="A framework for profiling ML inference across different tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/ml_inference_profiler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)