#!/usr/bin/env python3
"""
Ablation Study: Feature Importance Analysis
Analyzes which features contribute most to tau predictions
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
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neural_network import AdaptiveTauNetwork
from feature_extractor import WaypointFeatureExtractor

def analyze_feature_importance():
    """Analyze the importance of each feature for tau prediction"""
    
    print("="*60)
    print("ABLATION STUDY: Feature Importance Analysis")
    print("="*60)
    
    # Feature names
    feature_names = [
        # Geometric features (7)
        'min_clearance',
        'avg_clearance_1m', 
        'avg_clearance_2m',
        'passage_width',
        'obstacle_density_1m',
        'obstacle_density_2m',
        'num_obstacles_nearby',
        # Noise-specific features (3)
        'transparency_indicator',
        'occlusion_ratio',
        'position_uncertainty',
        # Path context features (5)
        'path_progress',
        'distance_to_goal',
        'path_curvature',
        'velocity',
        'heading_change',
        # Environment type features (5)
        'is_corridor',
        'is_open_space',
        'is_near_corner',
        'is_near_doorway',
        'boundary_distance'
    ]
    
    # Load model and test data
    checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AdaptiveTauNetwork(input_dim=20, hidden_dims=[128, 64, 32])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    with open('data/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Limit to subset for speed
    test_subset = test_data[:1000]
    
    print(f"\nAnalyzing {len(test_subset)} test samples...")
    
    # 1. Feature Statistics
    print("\n" + "="*40)
    print("1. Feature Statistics")
    print("="*40)
    
    feature_values = {name: [] for name in feature_names}
    tau_values = []
    
    with torch.no_grad():
        for sample in tqdm(test_subset, desc="Extracting features"):
            features = sample['features']
            for i, name in enumerate(feature_names):
                feature_values[name].append(features[i])
            
            # Get tau prediction
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            tau = model(features_tensor).item()
            tau_values.append(tau)  # Don't clip - let tau vary naturally
    
    # Calculate statistics
    feature_stats = {}
    for name in feature_names:
        values = feature_values[name]
        feature_stats[name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        print(f"{name:25s}: mean={feature_stats[name]['mean']:7.3f}, "
              f"std={feature_stats[name]['std']:7.3f}")
    
    # 2. Feature-Tau Correlation
    print("\n" + "="*40)
    print("2. Feature-Tau Correlation Analysis")
    print("="*40)
    
    correlations = {}
    for name in feature_names:
        # Pearson correlation
        pearson_corr, pearson_p = pearsonr(feature_values[name], tau_values)
        # Spearman correlation (for non-linear relationships)
        spearman_corr, spearman_p = spearmanr(feature_values[name], tau_values)
        
        correlations[name] = {
            'pearson': pearson_corr,
            'pearson_p': pearson_p,
            'spearman': spearman_corr,
            'spearman_p': spearman_p
        }
    
    # Sort by absolute correlation
    sorted_features = sorted(correlations.items(), 
                           key=lambda x: abs(x[1]['spearman']), 
                           reverse=True)
    
    print("\nTop 10 Most Correlated Features (Spearman):")
    for name, corr in sorted_features[:10]:
        print(f"{name:25s}: r={corr['spearman']:7.3f} (p={corr['spearman_p']:.3e})")
    
    # 3. Leave-One-Out Feature Importance
    print("\n" + "="*40)
    print("3. Leave-One-Out Feature Importance")
    print("="*40)
    
    baseline_predictions = []
    with torch.no_grad():
        for sample in test_subset:
            features = torch.FloatTensor(sample['features']).unsqueeze(0).to(device)
            tau = model(features).item()
            baseline_predictions.append(tau)
    
    feature_importance = {}
    
    for feature_idx, feature_name in enumerate(tqdm(feature_names, 
                                                    desc="Computing importance")):
        masked_predictions = []
        
        with torch.no_grad():
            for sample in test_subset:
                # Mask out the feature (set to mean value)
                features = sample['features'].copy()
                features[feature_idx] = feature_stats[feature_name]['mean']
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                tau = model(features_tensor).item()
                masked_predictions.append(tau)
        
        # Calculate importance as change in predictions
        importance = np.mean(np.abs(np.array(baseline_predictions) - 
                                   np.array(masked_predictions)))
        feature_importance[feature_name] = importance
    
    # Sort by importance
    sorted_importance = sorted(feature_importance.items(), 
                             key=lambda x: x[1], 
                             reverse=True)
    
    print("\nFeature Importance (by masking):")
    for name, importance in sorted_importance:
        print(f"{name:25s}: {importance:7.4f}")
    
    # 4. Visualization
    os.makedirs('results/ablations', exist_ok=True)
    
    # Plot correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Correlation bar plot
    names = [n[:15] for n, _ in sorted_features]  # Truncate names
    correlations_values = [c['spearman'] for _, c in sorted_features]
    
    axes[0].barh(range(len(names)), correlations_values)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Spearman Correlation with Tau')
    axes[0].set_title('Feature-Tau Correlation')
    axes[0].grid(True, alpha=0.3)
    
    # Importance bar plot
    names_imp = [n[:15] for n, _ in sorted_importance]
    importance_values = [i for _, i in sorted_importance]
    
    axes[1].barh(range(len(names_imp)), importance_values)
    axes[1].set_yticks(range(len(names_imp)))
    axes[1].set_yticklabels(names_imp, fontsize=8)
    axes[1].set_xlabel('Importance Score')
    axes[1].set_title('Feature Importance (Leave-One-Out)')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Analysis')
    plt.tight_layout()
    plt.savefig('results/ablations/feature_importance.png', dpi=150)
    plt.show()
    
    # 5. Feature Group Analysis
    print("\n" + "="*40)
    print("4. Feature Group Analysis")
    print("="*40)
    
    feature_groups = {
        'Geometric': feature_names[0:7],
        'Noise-specific': feature_names[7:10],
        'Path-context': feature_names[10:15],
        'Environment': feature_names[15:20]
    }
    
    group_importance = {}
    for group_name, group_features in feature_groups.items():
        # Average importance of features in group
        group_imp = np.mean([feature_importance[f] for f in group_features])
        group_importance[group_name] = group_imp
        print(f"{group_name:15s}: {group_imp:.4f}")
    
    # 6. Feature Interaction Analysis
    print("\n" + "="*40)
    print("5. Top Feature Pairs (Interaction)")
    print("="*40)
    
    # Analyze top 5 most important features
    top_features = [name for name, _ in sorted_importance[:5]]
    
    # Create scatter plots for top feature pairs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for i, feat1 in enumerate(top_features[:3]):
        for j, feat2 in enumerate(top_features[:3]):
            if i < j and plot_idx < 6:
                feat1_idx = feature_names.index(feat1)
                feat2_idx = feature_names.index(feat2)
                
                x = [s['features'][feat1_idx] for s in test_subset[:200]]
                y = [s['features'][feat2_idx] for s in test_subset[:200]]
                
                # Color by tau value
                colors = tau_values[:200]
                
                scatter = axes[plot_idx].scatter(x, y, c=colors, cmap='coolwarm', 
                                                alpha=0.6, s=10)
                axes[plot_idx].set_xlabel(feat1[:12])
                axes[plot_idx].set_ylabel(feat2[:12])
                axes[plot_idx].set_title(f'{feat1[:10]} vs {feat2[:10]}')
                plt.colorbar(scatter, ax=axes[plot_idx], label='Tau')
                
                plot_idx += 1
    
    plt.suptitle('Feature Interactions (colored by Tau)')
    plt.tight_layout()
    plt.savefig('results/ablations/feature_interactions.png', dpi=150)
    plt.show()
    
    # Save results
    results = {
        'feature_stats': feature_stats,
        'correlations': correlations,
        'feature_importance': feature_importance,
        'group_importance': group_importance,
        'top_features': top_features
    }
    
    import json
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_json = convert_numpy(results)
    
    with open('results/ablations/feature_importance_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to results/ablations/feature_importance_results.json")
    
    return results


if __name__ == "__main__":
    results = analyze_feature_importance()