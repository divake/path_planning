#!/usr/bin/env python3
"""
Create professional nonconformity score distribution plot for publication.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# Set publication quality parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.titlesize'] = 18
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5

def create_professional_nonconformity_plot():
    """Create professional nonconformity distribution plot using actual data."""

    # Load the actual nonconformity scores
    with open('results/ablations/nonconformity_scores.json', 'r') as f:
        nonconformity_scores = json.load(f)

    # Create single square figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Add thick black border frame
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2.0)

    # Create histogram with professional styling - simple blue bars
    n, bins, patches = ax.hist(nonconformity_scores, bins=40,
                               alpha=0.7, color='#2196F3',  # Material blue
                               edgecolor='black', linewidth=0.5)

    # Add q_hat line (90th percentile)
    ax.axvline(x=0, color='#D32F2F', linestyle='--', linewidth=2.0,
               label='q̂ (90th percentile)', alpha=0.9)

    # Clean styling - no title for professional papers
    ax.set_xlabel('Nonconformity Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)

    # Bold tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(10)

    # Set x-axis limits
    ax.set_xlim(-1.5, 0.2)

    # Simple legend for q_hat only - position to the left to avoid overlap
    legend = ax.legend(loc='upper left', frameon=True, fancybox=False,
                      shadow=False, borderpad=0.8, prop={'weight': 'bold', 'size': 10})
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(1.0)

    # Print statistics for verification
    print("\nNonconformity Score Statistics:")
    print(f"Total samples: {len(nonconformity_scores)}")
    print(f"Mean: {np.mean(nonconformity_scores):.3f}")
    print(f"Std: {np.std(nonconformity_scores):.3f}")
    print(f"Min: {np.min(nonconformity_scores):.3f}")
    print(f"Max: {np.max(nonconformity_scores):.3f}")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    save_path = 'results/ablations/nonconformity_professional.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved professional nonconformity plot: {save_path}")

    # Also save as PDF for publication
    save_path_pdf = save_path.replace('.png', '.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF version: {save_path_pdf}")

    plt.show()

if __name__ == "__main__":
    create_professional_nonconformity_plot()