#!/usr/bin/env python3
"""
Create professional feature importance plot for publication - exact replica with styling.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import rcParams

# Set font parameters for publication
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Times']
rcParams['font.size'] = 18
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 22
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['legend.fontsize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5

def create_professional_feature_importance(top_n=8):
    """Create professional feature importance plot - showing only top features using Leave-One-Out analysis.

    Args:
        top_n: Number of top features to show
    """

    # Load the data
    with open('results/ablations/feature_importance_results.json', 'r') as f:
        data = json.load(f)

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
        'dist_to_goal',
        'velocity',
        'boundary_distance',
        'is_near_corner',
        'trans_indicator',
        'heading_change',
        'pos_uncertertain',
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

    # Create single figure with square configuration
    fig, ax = plt.subplots(figsize=(10, 8))

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Add thick black border frame
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)

    # Feature Importance (Leave-One-Out)
    # Sort by importance score (as in original)
    sorted_indices_imp = sorted(range(len(feature_keys)),
                               key=lambda i: importance[feature_keys[i]],
                               reverse=True)

    # Take only top N features
    sorted_feature_names_imp = [feature_names[i] for i in sorted_indices_imp[:top_n]]
    sorted_feature_keys_imp = [feature_keys[i] for i in sorted_indices_imp[:top_n]]

    # Get importance values
    imp_values = [importance[key] for key in sorted_feature_keys_imp]

    # Create horizontal bar plot with larger bars
    y_pos = np.arange(len(sorted_feature_names_imp))
    bars = ax.barh(y_pos, imp_values, color='#2196F3', edgecolor='black', linewidth=1.5, height=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_feature_names_imp, fontsize=24, fontweight='bold')
    ax.set_xlabel('Feature Attribution', fontsize=26, fontweight='bold')

    # Set x-axis ticks to show fewer values with better spacing
    max_val = max(imp_values)
    tick_interval = 0.01
    ax.set_xticks([0, tick_interval, tick_interval*2, tick_interval*3])

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Set tick parameters with larger fonts
    ax.tick_params(axis='x', which='major', labelsize=20, width=1.5, length=5)
    ax.tick_params(axis='y', which='major', labelsize=24, width=1.5, length=5)

    # Bold axis labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(20)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(24)

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