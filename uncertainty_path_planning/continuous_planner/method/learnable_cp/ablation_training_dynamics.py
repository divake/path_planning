#!/usr/bin/env python3
"""
Ablation Study: Training Dynamics Analysis
Analyzes how the neural network learns and how tau evolves during training
"""

import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neural_network import AdaptiveTauNetwork

def analyze_training_dynamics():
    """Analyze the training dynamics from saved checkpoints"""
    
    print("="*60)
    print("ABLATION STUDY: Training Dynamics")
    print("="*60)
    
    # Load the saved checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\nBest model from epoch {checkpoint['epoch']} loaded")
    
    # Extract training history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        # Create visualizations directory
        os.makedirs('results/ablations', exist_ok=True)
        
        # Plot 1: Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training vs Validation Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='orange')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Coverage over epochs
        axes[0, 1].plot(history['coverage'], color='green')
        axes[0, 1].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].set_title('Coverage Guarantee Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Tau evolution
        axes[1, 0].plot(history['avg_tau'], color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Average Tau (m)')
        axes[1, 0].set_title('Average Tau Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Tau range (min to max)
        epochs = range(len(history['avg_tau']))
        axes[1, 1].fill_between(epochs, 
                                [t - 0.1 for t in history['avg_tau']], 
                                [t + 0.1 for t in history['avg_tau']], 
                                alpha=0.3, color='blue')
        axes[1, 1].plot(history['avg_tau'], color='blue', label='Mean Tau')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Tau Range (m)')
        axes[1, 1].set_title('Tau Variation During Training')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Training Dynamics Analysis')
        plt.tight_layout()
        plt.savefig('results/ablations/training_dynamics.png', dpi=150)
        plt.show()
        
        # Print statistics
        print("\nTraining Statistics:")
        print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"Final Coverage: {history['coverage'][-1]:.3f}")
        print(f"Final Avg Tau: {history['avg_tau'][-1]:.3f}m")
    
    # Analyze tau predictions on test data
    print("\n" + "="*40)
    print("Analyzing Tau Predictions on Test Data")
    print("="*40)
    
    # Load test data
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveTauNetwork(input_dim=20, hidden_dims=[128, 64, 32])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Collect tau predictions
    tau_by_noise = {'transparency': [], 'occlusion': [], 'localization': [], 'combined': []}
    tau_by_env = {}
    feature_tau_correlation = {}
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Analyzing test samples"):
            features = torch.FloatTensor(sample['features']).unsqueeze(0).to(device)
            tau = model(features).item()
            
            # Clamp tau to reasonable range
            # Don't clip tau - let it vary naturally for nonconformity scores
            
            noise_type = sample.get('noise_type', 'unknown')
            env_name = sample.get('env_name', 'unknown')
            
            if noise_type in tau_by_noise:
                tau_by_noise[noise_type].append(tau)
            
            if env_name not in tau_by_env:
                tau_by_env[env_name] = []
            tau_by_env[env_name].append(tau)
    
    # Plot tau distributions by noise type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot by noise type
    noise_data = []
    noise_labels = []
    for noise_type, taus in tau_by_noise.items():
        if taus:  # Only add if we have data
            noise_data.append(taus)
            noise_labels.append(noise_type)
    
    if noise_data:
        axes[0].boxplot(noise_data, labels=noise_labels)
        axes[0].set_xlabel('Noise Type')
        axes[0].set_ylabel('Tau (m)')
        axes[0].set_title('Tau Distribution by Noise Type')
        axes[0].grid(True, alpha=0.3)
    
    # Box plot by environment
    env_data = []
    env_labels = []
    for env_name, taus in tau_by_env.items():
        if taus:  # Only add if we have data
            env_data.append(taus)
            env_labels.append(env_name.replace('office', 'off'))
    
    if env_data:
        axes[1].boxplot(env_data, labels=env_labels)
        axes[1].set_xlabel('Environment')
        axes[1].set_ylabel('Tau (m)')
        axes[1].set_title('Tau Distribution by Environment')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Tau Predictions Analysis')
    plt.tight_layout()
    plt.savefig('results/ablations/tau_distributions.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("\nTau Statistics by Noise Type:")
    for noise_type, taus in tau_by_noise.items():
        if taus:
            print(f"{noise_type:15s}: mean={np.mean(taus):7.3f}, std={np.std(taus):7.3f}, "
                  f"min={np.min(taus):7.3f}, max={np.max(taus):7.3f}")
    
    print("\nTau Statistics by Environment:")
    for env_name, taus in tau_by_env.items():
        if taus:
            print(f"{env_name:15s}: mean={np.mean(taus):7.3f}, std={np.std(taus):7.3f}, "
                  f"samples={len(taus)}")
    
    # Analyze nonconformity score distribution
    print("\n" + "="*40)
    print("Nonconformity Score Analysis")
    print("="*40)
    
    # Calculate actual nonconformity scores
    nonconformity_scores = []
    for sample in test_data[:500]:  # Use subset for speed
        features = torch.FloatTensor(sample['features']).unsqueeze(0).to(device)
        tau = model(features).item()
        clearance = sample['clearance']
        
        # Nonconformity score = tau - clearance
        # If positive: conservative (safe)
        # If negative: aggressive (potentially unsafe)
        score = tau - clearance
        nonconformity_scores.append(score)
    
    # Plot nonconformity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(nonconformity_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Safety Boundary')
    plt.xlabel('Nonconformity Score')
    plt.ylabel('Frequency')
    plt.title('Nonconformity Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/ablations/nonconformity_distribution.png', dpi=150)
    plt.show()
    
    # Calculate coverage
    safe_predictions = sum(1 for s in nonconformity_scores if s >= 0)
    coverage = safe_predictions / len(nonconformity_scores)
    print(f"Empirical Coverage: {coverage:.3f} (Target: 0.900)")
    print(f"Safe Predictions: {safe_predictions}/{len(nonconformity_scores)}")
    
    # Save results
    results = {
        'tau_by_noise': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                        for k, v in tau_by_noise.items() if v},
        'tau_by_env': {k: {'mean': np.mean(v), 'std': np.std(v), 'count': len(v)} 
                      for k, v in tau_by_env.items() if v},
        'coverage': coverage,
        'nonconformity_stats': {
            'mean': np.mean(nonconformity_scores),
            'std': np.std(nonconformity_scores),
            'min': np.min(nonconformity_scores),
            'max': np.max(nonconformity_scores)
        }
    }
    
    import json
    with open('results/ablations/training_dynamics_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results/ablations/training_dynamics_results.json")
    
    return results


if __name__ == "__main__":
    results = analyze_training_dynamics()