#!/usr/bin/env python3
"""
Quick training script using existing datasets
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_network import AdaptiveTauNetwork
from loss_function import AdaptiveTauLoss


class SimpleDataset(Dataset):
    """Simple dataset for training"""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'features': torch.FloatTensor(sample['features']),
            'clearance': torch.FloatTensor([sample['clearance']]),
            'noise_type': sample.get('noise_type', 'unknown'),
            'env_name': sample.get('env_name', 'unknown')
        }


def train_model():
    """Train the model using existing datasets"""
    
    # Load datasets
    print("Loading datasets...")
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create data loaders
    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AdaptiveTauNetwork(
        input_dim=20,
        hidden_dims=[128, 64, 32],
        dropout_rate=0.2
    ).to(device)
    
    # Loss and optimizer
    criterion = AdaptiveTauLoss(
        safety_weight=10.0,
        efficiency_weight=1.0,
        smoothness_weight=0.5,
        coverage_weight=2.0
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    num_epochs = 20  # Quick training
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = batch['features'].to(device)
            clearances = batch['clearance'].to(device)
            
            # Forward pass
            predicted_tau = model(features).squeeze()
            
            # Loss
            loss, loss_dict = criterion(
                predicted_tau, clearances.squeeze(), features
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        coverages = []
        avg_taus = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                clearances = batch['clearance'].to(device)
                
                predicted_tau = model(features).squeeze()
                loss, loss_dict = criterion(
                    predicted_tau, clearances.squeeze(), features
                )
                
                val_losses.append(loss.item())
                coverages.append(loss_dict['empirical_coverage'])
                avg_taus.append(loss_dict['avg_predicted_tau'])
        
        # Metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_coverage = np.mean(coverages)
        avg_tau = np.mean(avg_taus)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Coverage={avg_coverage:.3f}, "
              f"Avg Tau={avg_tau:.3f}m")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_loss': best_val_loss
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print(f"Saved best model with val_loss={best_val_loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Test on some samples
    print("\nTesting on a few samples:")
    model.eval()
    with torch.no_grad():
        for i in range(3):
            sample = val_data[i]
            features = torch.FloatTensor(sample['features']).unsqueeze(0).to(device)
            tau = model(features).item()
            clearance = sample['clearance']
            print(f"Sample {i+1}: Predicted tau={tau:.3f}m, Actual clearance={clearance:.3f}m, "
                  f"Safe={tau >= clearance}")
    
    return model


if __name__ == "__main__":
    model = train_model()
    print("\nModel training complete! Checkpoint saved to checkpoints/best_model.pth")