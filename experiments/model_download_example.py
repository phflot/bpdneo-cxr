#!/usr/bin/env python
"""
Model download and inference example for BPDneo-CXR.
Demonstrates loading a pre-trained model and running inference on an example X-ray image.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from bpd_torch.models.model_util import (
    load_pretrained_model, 
    get_preprocessing_transforms,
    list_available_models
)


def main():

    print("="*70)
    print("BPDneo-CXR Model Download and Inference Example")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nAvailable models:")
    print("-" * 50)
    models = list_available_models()
    for name, info in models.items():
        print(f"  {name}: AUROC={info['auroc']:.3f}")
    
    model_name = "bpd_xrv_progfreeze_lp_cutmix"
    print(f"\nLoading model: {model_name}")
    
    model = load_pretrained_model(model_name, device=device)
    print("- Model loaded successfully")
    
    image_path = Path("../img/x-ray_example.png")
    print(f"\nLoading example image from: {image_path}")
    img = Image.open(image_path)
    print(f"- Image loaded (size: {img.size}, mode: {img.mode})")
    
    print("\nApplying preprocessing transforms...")
    transform = get_preprocessing_transforms(model_name)
    
    img_tensor = transform(img)
    print(f"- Image preprocessed to tensor shape: {img_tensor.shape}")
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    print(f"- Batch prepared: {img_batch.shape}")
    
    print("\nRunning inference...")
    with torch.no_grad():
        logits = model(img_batch)
        print(f"  Raw logits: {logits.item():.4f}")
        
        prob = torch.sigmoid(logits).item()
        print(f"  Probability of moderate/severe BPD: {prob:.4f}")
        
        prediction = "Moderate/Severe BPD" if prob > 0.5 else "No/Mild BPD"
        print(f"  Prediction: {prediction}")
    
    print("\n" + "="*70)
    print("Inference pipeline completed successfully!")
    print("="*70)
    
    print("\nModel Information:")
    print("-" * 50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model on device: {next(model.parameters()).device}")


if __name__ == "__main__":
    exit(main())