#!/usr/bin/env python3
"""
Example usage of the FastMRI knee dataset

This script demonstrates how to use the FastMRIH5SliceDataset for training
or evaluation with the CoSIGN framework.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/cosign/cosign')

from datasets.MRI_datasets_knee_kspace import FastMRIH5SliceDataset, create_dataloader
import torch
from torch.utils.data import DataLoader

# Example 1: Direct dataset usage
def example_direct_dataset():
    """Example of using FastMRIH5SliceDataset directly"""
    print("=== Example 1: Direct Dataset Usage ===")
    
    # Create dataset instance
    dataset = FastMRIH5SliceDataset(
        root='/path/to/fastmri/knee/data',  # Replace with actual path
        is_complex=False,
        key='reconstruction_rss',
        target_shape=(256, 256),
        sort=True
    )
    
    print(f"Dataset size: {len(dataset)} slices")
    
    # Create a simple DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Process a batch
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("Shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        break  # Just show first batch
    
    return dataset

# Example 2: Using the config-based approach
def example_config_based():
    """Example of using create_dataloader with config object"""
    print("\n=== Example 2: Config-based Usage ===")
    
    # Mock config object (replace with actual config loading)
    class Config:
        def __init__(self):
            self.data = DataConfig()
            self.training = TrainingConfig()
    
    class DataConfig:
        def __init__(self):
            self.root = '/path/to/fastmri/knee/data'
            self.is_complex = True
            self.h5_key = 'reconstruction_rss'
            self.image_size = 256
    
    class TrainingConfig:
        def __init__(self):
            self.batch_size = 8
            self.num_workers = 4
    
    config = Config()
    
    # Create dataloaders
    train_loader, val_loader = create_dataloader(
        config, 
        evaluation=False, 
        sort=True
    )
    
    print(f"Created train and validation loaders")
    
    # Process a training batch
    for batch in train_loader:
        print("Training batch keys:", batch.keys())
        print("Training batch shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        break
    
    return train_loader, val_loader

# Example 3: Integration with existing framework
def example_framework_integration():
    """Example of using the dataset with cc.image_datasets"""
    print("\n=== Example 3: Framework Integration ===")
    
    try:
        from cc.image_datasets import load_fastmri
        
        # Use the integrated loader
        loader_generator = load_fastmri(
            data_dir='/path/to/fastmri/knee/data',
            batch_size=4,
            image_size=256,
            is_complex=True,
            h5_key='reconstruction_rss',
            deterministic=False
        )
        
        # Get a batch
        batch = next(loader_generator)
        print("Framework batch keys:", batch.keys())
        print("Framework batch shapes:")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
        
        return loader_generator
        
    except ImportError as e:
        print(f"Framework integration requires MPI: {e}")
        return None

def main():
    """Run all examples"""
    print("FastMRI Knee Dataset Usage Examples")
    print("=" * 50)
    
    # Note: These examples assume you have actual FastMRI data
    # For testing, you would replace paths with actual data directories
    
    try:
        # Example 1: Direct usage
        dataset = example_direct_dataset()
        
        # Example 2: Config-based usage  
        train_loader, val_loader = example_config_based()
        
        # Example 3: Framework integration
        framework_loader = example_framework_integration()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo use with real data:")
        print("1. Replace '/path/to/fastmri/knee/data' with your actual data directory")
        print("2. Ensure your .h5 files have 'reconstruction_rss' and 'kspace' keys")
        print("3. Adjust batch_size and num_workers based on your hardware")
        
    except Exception as e:
        print(f"\n⚠️  Examples require actual FastMRI data: {e}")
        print("This is normal if you don't have FastMRI data available.")

if __name__ == "__main__":
    main()