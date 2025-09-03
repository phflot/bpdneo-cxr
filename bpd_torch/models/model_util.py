"""
Model utility functions for downloading and loading pre-trained BPD prediction models.
"""

import torch
import os
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import json
import numpy as np
from PIL import Image


MODEL_CONFIGS = {
    "bpd_xrv_progfreeze_lp_cutmix": {
        "file_id": "nLYMSE8jRSg3j8j",
        "description": "Best performing model with XRV pretraining, progressive freezing, linear probing, and CutMix",
        "auroc": 0.783,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": ["layer1", "layer2", "layer3"]
    },
    "bpd_xrv_progfreeze": {
        "file_id": "SRxGJzLSpEMMAD4",
        "description": "Baseline with XRV pretraining and progressive freezing (no augmentation)",
        "auroc": 0.775,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": ["layer1", "layer2", "layer3"]
    },
    "bpd_rgb_progfreeze": {
        "file_id": "W7EmnFDSFwoFSBL",
        "description": "ImageNet baseline with progressive freezing (for comparison)",
        "auroc": 0.717,
        "preprocessing": "imagenet",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "torchvision",
        "frozen_layers": ["layer1", "layer2", "layer3"]
    },
    "bpd_xrv_fullft": {
        "file_id": "w2czAo4oYxFaAGi",
        "description": "XRV pretraining with full fine-tuning (no freezing)",
        "auroc": 0.761,
        "preprocessing": "xrv",
        "input_size": 512,
        "num_classes": 1,
        "architecture": "resnet50",
        "backbone": "xrv",
        "frozen_layers": []
    }
}


class ModelDownloader:
    """Downloads pre-trained BPD prediction models."""
    
    def __init__(self, model_name: str, save_dir: str = "~/.bpdneo/models"):
        """
        Initialize the model downloader.
        
        Args:
            model_name: Name of the model to download (see MODEL_CONFIGS keys)
            save_dir: Directory to save downloaded models
        """
        if model_name not in MODEL_CONFIGS:
            available_models = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )
        
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.save_dir = Path(os.path.expanduser(save_dir))
        self.model_path = self.save_dir / f"{model_name}.pth"
        self.config_path = self.save_dir / f"{model_name}_config.json"
        
        # Download URL
        file_id = self.config["file_id"]
        self.model_url = f"https://cloud.hiz-saarland.de/public.php/dav/files/{file_id}"
    
    def download_model(self) -> Path:
        """
        Download the model weights if not already present.
        
        Returns:
            Path to the downloaded model weights
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config file
        if not self.config_path.exists():
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        
        # Check if model already exists
        if self.model_path.exists():
            print(f"Model '{self.model_name}' already exists at {self.model_path}")
            return self.model_path
        
        print(f"Downloading {self.model_name}...")
        print(f"Description: {self.config['description']}")
        print(f"Expected AUROC: {self.config['auroc']}")
        
        try:
            response = requests.get(self.model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                with tqdm(
                    desc=f"Downloading {self.model_name}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Model downloaded successfully to {self.model_path}")
            return self.model_path
            
        except requests.exceptions.RequestException as e:
            if self.model_path.exists():
                self.model_path.unlink()  # Remove partial download
            raise RuntimeError(f"Failed to download model: {e}")


def download_model_weights(model_name: str, save_dir: str = "~/.bpdneo/models") -> Path:
    """
    Download pre-trained model weights.
    
    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the model
    
    Returns:
        Path to the downloaded model weights
    """
    downloader = ModelDownloader(model_name, save_dir)
    return downloader.download_model()


def _build_model_from_config(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Build model architecture from configuration, matching training setup.
    """
    import torchxrayvision as xrv
    from torchvision import models as tvm
    from bpd_torch.models.model import BPDModel
    
    if config["backbone"] == "xrv":
        base = xrv.models.ResNet(weights="resnet50-res512-all")
    elif config["backbone"] == "torchvision":
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")
    
    return BPDModel(base)


def load_pretrained_model(
    model_name: str,
    device: Optional[torch.device] = None,
    save_dir: str = "~/.bpdneo/models"
) -> torch.nn.Module:
    """
    Load a pre-trained BPD prediction model.
    
    Args:
        model_name: Name of the model to load
        device: Device to load the model on (default: cuda if available, else cpu)
        save_dir: Directory where models are saved
    
    Returns:
        Loaded PyTorch model ready for inference
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download model if needed
    model_path = download_model_weights(model_name, save_dir)
    
    # Load configuration
    config = MODEL_CONFIGS[model_name]
    
    # Build model with correct backbone and wrapper
    model = _build_model_from_config(config).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Loaded model '{model_name}' on {device}")
    print(f"Preprocessing: {config['preprocessing']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")
    
    return model


def get_preprocessing_transforms(model_name: str):
    """
    Get the appropriate preprocessing transforms for a model.
    Matches the exact preprocessing used during training.
    
    Args:
        model_name: Name of the model
    
    Returns:
        torchvision transforms for preprocessing
    """
    from torchvision import transforms as T
    import torchxrayvision as xrv
    
    config = MODEL_CONFIGS[model_name]
    input_size = config["input_size"]
    
    if config["preprocessing"] == "xrv":
        # XRV preprocessing as used in training:
        # Load grayscale -> xrv.datasets.normalize -> tensor -> resize(512)
        def _xrv_transform(pil_img):
            """Apply XRV preprocessing matching training."""
            # Convert to grayscale numpy array
            arr = np.array(pil_img.convert('L'))
            # Apply XRV normalization (expects values 0-255)
            arr = xrv.datasets.normalize(arr, 255)
            # Convert to tensor (this handles the conversion to float and adds channel dim)
            tensor = T.ToTensor()(arr)
            # Resize to target size
            resize = T.Resize((input_size, input_size), antialias=True)
            return resize(tensor)
        
        transform = T.Lambda(_xrv_transform)
        
    elif config["preprocessing"] == "imagenet":
        # ImageNet preprocessing for RGB models
        transform = T.Compose([
            T.Resize((input_size, input_size), antialias=True),
            T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown preprocessing type: {config['preprocessing']}")
    
    return transform


def list_available_models() -> Dict[str, Any]:
    """
    List all available pre-trained models with their descriptions.
    
    Returns:
        Dictionary of model names and their configurations
    """
    models_info = {}
    for name, config in MODEL_CONFIGS.items():
        models_info[name] = {
            "description": config["description"],
            "auroc": config["auroc"],
            "architecture": config["architecture"],
            "preprocessing": config["preprocessing"]
        }
    return models_info


if __name__ == "__main__":
    # Example usage
    print("Available BPD prediction models:")
    print("-" * 50)
    
    for name, info in list_available_models().items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  AUROC: {info['auroc']}")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Preprocessing: {info['preprocessing']}")
    
    print("\n" + "-" * 50)
    print("\nTo download and use a model:")
    print(">>> from bpd_torch.models.model_util import load_pretrained_model")
    print(">>> model = load_pretrained_model('bpd_xrv_progfreeze_lp_cutmix')")