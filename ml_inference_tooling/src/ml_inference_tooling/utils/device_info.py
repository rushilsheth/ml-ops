import os
import platform
import subprocess
import json
from typing import Dict, Any, List, Optional

class DeviceInfo:
    """
    Utility for gathering system information.
    """
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """
        Get CPU information.
        
        Returns:
            Dictionary with CPU specs
        """
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "python_version": platform.python_version(),
        }
        
        try:
            import psutil
            info.update({
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            })
        except ImportError:
            pass
            
        return info
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """
        Get GPU information.
        
        Returns:
            List of dictionaries with GPU specs
        """
        gpus = []
        
        # Try PyTorch approach
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / (1024**2),
                        "cuda_version": torch.version.cuda,
                        "source": "pytorch"
                    })
        except (ImportError, Exception):
            pass
        
        # Try TensorFlow approach if PyTorch didn't work
        if not gpus:
            try:
                import tensorflow as tf
                devices = tf.config.list_physical_devices('GPU')
                for i, dev in enumerate(devices):
                    gpus.append({
                        "index": i,
                        "name": dev.name,
                        "source": "tensorflow"
                    })
            except (ImportError, Exception):
                pass
                
        # Try nvidia-smi as a last resort
        if not gpus:
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
                output = output.decode("utf-8").strip()
                
                for line in output.split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": float(parts[2]),
                            "driver_version": parts[3],
                            "source": "nvidia-smi"
                        })
            except (subprocess.SubprocessError, Exception):
                pass
                
        return gpus
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        Get complete system information.
        
        Returns:
            Dictionary with system specs
        """
        return {
            "cpu": DeviceInfo.get_cpu_info(),
            "gpus": DeviceInfo.get_gpu_info(),
            "timestamp": DeviceInfo.get_timestamp()
        }
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp.
        
        Returns:
            ISO format timestamp
        """
        from datetime import datetime
        return datetime.now().isoformat()
    
    @staticmethod
    def save_system_info(output_path: str) -> None:
        """
        Save system information to a file.
        
        Args:
            output_path: Path to save the system info
        """
        info = DeviceInfo.get_system_info()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"System information saved to {output_path}")
    
    @staticmethod
    def print_summary() -> None:
        """
        Print a summary of system information.
        """
        info = DeviceInfo.get_system_info()
        
        print("\n===== SYSTEM INFORMATION =====")
        print(f"Platform: {info['cpu']['platform']}")
        print(f"Python Version: {info['cpu']['python_version']}")
        
        if 'physical_cores' in info['cpu']:
            print(f"CPU: {info['cpu']['processor']} ({info['cpu']['physical_cores']} cores, {info['cpu']['logical_cores']} threads)")
            print(f"RAM: {info['cpu']['memory_total_gb']} GB")
        else:
            print(f"CPU: {info['cpu']['processor']}")
        
        if info['gpus']:
            print("\nGPUs:")
            for gpu in info['gpus']:
                if 'memory_total_mb' in gpu:
                    print(f"  {gpu['name']} ({gpu['memory_total_mb'] / 1024:.1f} GB)")
                else:
                    print(f"  {gpu['name']}")
                    
            if 'cuda_version' in info['gpus'][0]:
                print(f"CUDA Version: {info['gpus'][0]['cuda_version']}")
            elif 'driver_version' in info['gpus'][0]:
                print(f"Driver Version: {info['gpus'][0]['driver_version']}")
        else:
            print("\nNo GPUs detected")
            
        print("===============================\n")
