#!/usr/bin/env python3
"""
Create professional feature importance plot for publication - exact replica with styling.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import rcParams

# Set font parameters for publication
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['axes.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5

def create_professional_feature_importance():
    """Create professional feature importance plot - exact replica."""

    # Load the data
    with open('results/ablations/feature_importance_results.json', 'r') as f:
        data = json.load(f)

    correlations = data['correlations']
    importance = data['feature_importance']

    # All feature names in exact order from original
    feature_names = [
        'passage_width',
        'min_clearance',
        'avg_clear_1m',
        'occlusion_ratio',
        'obs_density',  # truncated in original
        'avg_clear_2m',
        'obs_density',  # truncated in original (2m version)
        'is_open_space',
        'is_near_doorway',
        'num_obs_near',
        'path_progress',
        'is_corridor',
        'distance_to_goa',
        'velocity',
        'boundary_distan',
        'is_near_corner',
        'trans_indicator',
        'heading_change',
        'position_uncert',
        'path_curvature'
    ]

    # Map truncated names to actual keys
    feature_keys = [
        'passage_width',
        'min_clearance',
        'avg_clearance_1m',
        'occlusion_ratio',
        'obstacle_density_1m',
        'avg_clearance_2m',
        'obstacle_density_2m',
        'is_open_space',
        'is_near_doorway',
        'num_obstacles_nearby',
        'path_progress',
        'is_corridor',
        'distance_to_goal',
        'velocity',
        'boundary_distance',
        'is_near_corner',
        'transparency_indicator',
        'heading_change',
        'position_uncertainty',
        'path_curvature'
    ]

    # Create figure with square configuration
    fig = plt.figure(figsize=(14, 7))

    # Set white background
    fig.patch.set_facecolor('white')

    # Create subplots side by side
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    # Set white background and black borders for both
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.0)

    # LEFT PLOT: Feature-Tau Correlation
    # Sort features by absolute Spearman correlation (as in original)
    sorted_indices = sorted(range(len(feature_keys)),
                          key=lambda i: abs(correlations[feature_keys[i]]['spearman']
                                          if not np.isnan(correlations[feature_keys[i]]['spearman'])
                                          else 0),
                          reverse=True)

    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_feature_keys = [feature_keys[i] for i in sorted_indices]

    # Get correlation values (handle NaN for is_open_space)
    corr_values = []
    for key in sorted_feature_keys:
        val = correlations[key]['spearman']
        if np.isnan(val):
            val = 0.0  # Set NaN to 0
        corr_values.append(val)

    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_feature_names))
    bars1 = ax1.barh(y_pos, corr_values, color='#2196F3', edgecolor='black', linewidth=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_feature_names, fontsize=8)
    ax1.set_xlabel('Spearman Correlation with Tau', fontsize=12, fontweight='bold')
    # No subplot title for publication
    ax1.set_xlim(-1.0, 1.0)

    # Add vertical line at 0
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.0, alpha=0.5)

    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Bold axis labels
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(8)

    # RIGHT PLOT: Feature Importance (Leave-One-Out)
    # Sort by importance score (as in original)
    sorted_indices_imp = sorted(range(len(feature_keys)),
                               key=lambda i: importance[feature_keys[i]],
                               reverse=True)

    sorted_feature_names_imp = [feature_names[i] for i in sorted_indices_imp]
    sorted_feature_keys_imp = [feature_keys[i] for i in sorted_indices_imp]

    # Get importance values
    imp_values = [importance[key] for key in sorted_feature_keys_imp]

    # Create horizontal bar plot
    y_pos_imp = np.arange(len(sorted_feature_names_imp))
    bars2 = ax2.barh(y_pos_imp, imp_values, color='#2196F3', edgecolor='black', linewidth=0.5)

    ax2.set_yticks(y_pos_imp)
    ax2.set_yticklabels(sorted_feature_names_imp, fontsize=8)
    ax2.set_xlabel('Feature Attribution (Leave-One-Out)', fontsize=12, fontweight='bold')
    # No subplot title for publication

    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    # Bold axis labels
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(8)

    # No title for publication

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    save_path = 'results/ablations/feature_importance_professional.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved professional feature importance plot: {save_path}")

    # Also save as PDF for publication
    save_path_pdf = save_path.replace('.png', '.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved PDF version: {save_path_pdf}")

if __name__ == "__main__":
    create_professional_feature_importance()