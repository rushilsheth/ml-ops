import os
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

class DataLoader:
    """
    Utility for loading and preprocessing test data for inference.
    """
    
    def __init__(self, data_path: str = None, input_shape: List[int] = [1, 3, 224, 224], 
                 batch_size: int = 1, dtype: np.dtype = np.float32, normalize: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the dataset
            input_shape: Expected input shape
            batch_size: Batch size
            dtype: Data type
            normalize: Whether to normalize data
        """
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self.normalize = normalize
        self.data = None
        
    def load_random_data(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate random data for testing.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Random numpy array
        """
        bs = batch_size if batch_size is not None else self.batch_size
        shape = list(self.input_shape)
        shape[0] = bs
        
        data = np.random.rand(*shape).astype(self.dtype)
        
        if self.normalize and self.dtype == np.float32:
            data = (data - 0.5) * 2.0  # Normalize to [-1, 1]
            
        return data
    
    def load_numpy_data(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Load data from numpy file.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Numpy array
        """
        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        if self.data is None:
            self.data = np.load(self.data_path)
            
        bs = batch_size if batch_size is not None else self.batch_size
        
        # Handle different data shapes
        if len(self.data.shape) == len(self.input_shape):
            # Data already has batch dimension
            if bs <= self.data.shape[0]:
                return self.data[:bs].astype(self.dtype)
            else:
                # Repeat data to fill batch
                repeats = (bs + self.data.shape[0] - 1) // self.data.shape[0]
                repeated = np.repeat(self.data, repeats, axis=0)
                return repeated[:bs].astype(self.dtype)
        else:
            # Add batch dimension
            repeated = np.repeat(self.data[np.newaxis, ...], bs, axis=0)
            return repeated.astype(self.dtype)
            
    def load_image_folder(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Load images from a folder.
        
        Args:
            batch_size: Override default batch size
            
        Returns:
            Numpy array of images
        """
        try:
            from PIL import Image
            
            if not self.data_path or not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Image folder not found: {self.data_path}")
                
            bs = batch_size if batch_size is not None else self.batch_size
            image_files = [f for f in os.listdir(self.data_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                raise ValueError(f"No image files found in {self.data_path}")
                
            # Load and preprocess images
            images = []
            for i in range(bs):
                img_file = image_files[i % len(image_files)]
                img_path = os.path.join(self.data_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.input_shape[2], self.input_shape[1]))
                img_array = np.array(img).transpose(2, 0, 1)  # HWC to CHW
                images.append(img_array)
                
            data = np.stack(images).astype(self.dtype)
            
            if self.normalize and self.dtype == np.float32:
                data = data / 255.0  # Normalize to [0, 1]
                if self.normalize == "imagenet":
                    # ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                    data = (data - mean) / std
                
            return data
            
        except ImportError:
            print("PIL not installed. Please install with 'pip install pillow'.")
            raise