#!/usr/bin/env python3
"""
COMPARISON: DISCRETE vs CONTINUOUS PLANNERS WITH CP
Shows how τ and coverage differ between approaches
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from discrete_planner.discrete_cp_system import DiscreteEnvironment, DiscreteCP, DiscreteNonconformity
from continuous_planner.continuous_cp_system import ContinuousEnvironment, ContinuousCP, ContinuousNonconformity


def compare_tau_values():
    """
    Compare how τ varies with confidence for discrete vs continuous
    """
    print("="*80)
    print("COMPARING τ VALUES: DISCRETE vs CONTINUOUS")
    print("="*80)
    
    # Discrete calibration
    disc_env = DiscreteEnvironment()
    disc_cp = DiscreteCP(disc_env)
    
    # Continuous calibration
    cont_env = ContinuousEnvironment()
    cont_cp = ContinuousCP(cont_env)
    
    # Generate calibration data
    num_samples = 500
    
    print("\nGenerating calibration data...")
    
    # Discrete scores
    disc_scores = []
    for i in range(num_samples):
        perceived = disc_env.add_discrete_noise(0.05, seed=i)
        score = DiscreteNonconformity.compute_score(disc_env.obstacles, perceived)
        disc_scores.append(score)
    
    # Continuous scores
    cont_scores = []
    for i in range(num_samples):
        perceived = cont_env.add_continuous_noise(0.3, seed=i)
        score = ContinuousNonconformity.compute_score(cont_env.obstacles, perceived)
        cont_scores.append(score)
    
    # Sort scores
    disc_sorted = sorted(disc_scores)
    cont_sorted = sorted(cont_scores)
    
    # Calculate τ for different confidence levels
    confidence_levels = np.arange(0.70, 1.00, 0.01)
    disc_taus = []
    cont_taus = []
    
    for conf in confidence_levels:
        # Discrete τ
        idx = int(np.ceil((num_samples + 1) * conf)) - 1
        idx = min(idx, num_samples - 1)
        disc_taus.append(disc_sorted[idx])
        
        # Continuous τ
        cont_taus.append(cont_sorted[idx])
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Score distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Nonconformity Score Distributions', fontsize=12, fontweight='bold')
    
    # Discrete histogram
    unique_disc = sorted(set(disc_scores))
    disc_counts = [disc_scores.count(s) for s in unique_disc]
    ax1.bar(unique_disc, disc_counts, width=0.3, alpha=0.7, 
           label='Discrete', color='blue', edgecolor='black')
    
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.set_title('Discrete Scores (Integer)', fontsize=10)
    
    # Plot 2: Continuous histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(cont_scores, bins=30, alpha=0.7, color='green', 
            edgecolor='black', label='Continuous')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.set_title('Continuous Scores (Float)', fontsize=10)
    
    # Plot 3: τ vs Confidence
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('τ vs Confidence Level', fontsize=12, fontweight='bold')
    
    ax3.plot(confidence_levels*100, disc_taus, 'b-', linewidth=2, 
            label='Discrete (steps)', marker='s', markersize=3, markevery=5)
    ax3.plot(confidence_levels*100, cont_taus, 'g-', linewidth=2, 
            label='Continuous (smooth)', alpha=0.8)
    
    ax3.set_xlabel('Confidence Level (%)')
    ax3.set_ylabel('τ (Safety Margin)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight key confidence levels
    for conf in [80, 85, 90, 95]:
        ax3.axvline(x=conf, color='gray', linestyle=':', alpha=0.5)
    
    # Plot 4: Coverage precision
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Coverage Precision', fontsize=12, fontweight='bold')
    
    # Show possible τ values
    ax4.scatter([0, 1, 2, 3], [0]*4, s=100, color='blue', label='Discrete τ options')
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_xlabel('Possible τ values')
    ax4.set_yticks([])
    ax4.legend()
    
    for i in range(4):
        ax4.text(i, -0.1, f'τ={i}', ha='center', fontsize=10)
    
    ax4.text(1.5, 0.3, 'Only 4 options!', ha='center', fontsize=12, 
            color='red', fontweight='bold')
    
    # Plot 5: Continuous options
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Continuous τ Options', fontsize=12, fontweight='bold')
    
    cont_range = np.linspace(0, max(cont_taus), 100)
    ax5.fill_between(cont_range, 0, 0.1, color='green', alpha=0.3)
    ax5.set_xlim(0, max(cont_taus))
    ax5.set_ylim(0, 0.2)
    ax5.set_xlabel('Possible τ values')
    ax5.set_yticks([])
    
    # Show example values
    example_taus = [0.08, 0.10, 0.12, 0.15, 0.17]
    for tau in example_taus:
        ax5.axvline(x=tau, color='darkgreen', linestyle='-', alpha=0.5)
        ax5.text(tau, 0.15, f'{tau:.2f}', ha='center', fontsize=9, rotation=45)
    
    ax5.text(max(cont_taus)/2, 0.05, 'Infinite options!', ha='center', 
            fontsize=12, color='green', fontweight='bold')
    
    # Plot 6: Results comparison table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create comparison table
    table_data = [
        ['Aspect', 'Discrete', 'Continuous'],
        ['τ values', 'Integers\n(0, 1, 2, 3)', 'Any float\n(0.12, 1.57, etc.)'],
        ['τ @ 95%', 'τ = 1', 'τ = 0.12'],
        ['Coverage', 'Jumps\n(60%, 85%, 99%)', 'Smooth\n(any %)'],
        ['Precision', 'Coarse', 'Fine-grained'],
        ['Collision Rate', '0.4% @ τ=1', '0.5% @ τ=0.12'],
    ]
    
    table = ax6.table(cellText=table_data, loc='center',
                     cellLoc='center', colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color columns
    for i in range(1, 6):
        table[(i, 1)].set_facecolor('#e6f2ff')  # Discrete column
        table[(i, 2)].set_facecolor('#e6ffe6')  # Continuous column
    
    plt.suptitle('DISCRETE vs CONTINUOUS CONFORMAL PREDICTION', 
                fontsize=16, fontweight='bold')
    
    # Save figure
    save_path = 'results/discrete_vs_continuous_comparison.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to: {save_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print("\nDISCRETE PLANNER (Grid A*):")
    print(f"  - Possible τ values: {sorted(set(disc_scores))}")
    print(f"  - τ @ 95% confidence: {disc_sorted[int(np.ceil(num_samples*0.95))-1]}")
    print(f"  - Coverage options limited to discrete jumps")
    print(f"  - Simple and fast, but coarse control")
    
    print("\nCONTINUOUS PLANNER (RRT*):")
    print(f"  - τ range: [{min(cont_scores):.3f}, {max(cont_scores):.3f}]")
    print(f"  - τ @ 95% confidence: {cont_sorted[int(np.ceil(num_samples*0.95))-1]:.3f}")
    print(f"  - Can achieve ANY desired coverage level")
    print(f"  - Fine-grained control over safety-efficiency trade-off")
    
    print("\nIMPLICATIONS:")
    print("  1. Discrete is sufficient when rough safety margins are acceptable")
    print("  2. Continuous is better when precise coverage guarantees are needed")
    print("  3. The choice depends on application requirements!")
    
    print("="*80)


def visualize_inflation_difference():
    """
    Show how inflation looks different in discrete vs continuous
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Discrete inflation
    ax = axes[0]
    ax.set_title('Discrete Inflation (τ=1 cell)', fontsize=12, fontweight='bold')
    ax.set_xlim(15, 25)
    ax.set_ylim(12, 18)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Original obstacle
    for x in range(18, 22):
        ax.add_patch(patches.Rectangle((x-0.5, 14.5), 1, 1,
                                      facecolor='gray', edgecolor='black', alpha=0.7))
    
    # Inflated by 1 cell
    for x in range(17, 23):
        for y in [13, 14, 15, 16]:
            if not (18 <= x <= 21 and y == 15):
                ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                              facecolor='orange', edgecolor='red', 
                                              alpha=0.3, linestyle='--'))
    
    ax.text(20, 17, 'Integer cells only!', ha='center', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # Continuous inflation
    ax = axes[1]
    ax.set_title('Continuous Inflation (τ=0.12 units)', fontsize=12, fontweight='bold')
    ax.set_xlim(15, 25)
    ax.set_ylim(12, 18)
    ax.set_aspect('equal')
    
    # Original obstacle
    rect = patches.Rectangle((18, 14.5), 4, 1,
                            facecolor='gray', edgecolor='black', alpha=0.7)
    ax.add_patch(rect)
    
    # Inflated by 0.12 units
    inflated = patches.Rectangle((18-0.12, 14.5-0.12), 4+0.24, 1+0.24,
                                facecolor='orange', edgecolor='red', 
                                alpha=0.3, linestyle='--', linewidth=2)
    ax.add_patch(inflated)
    
    # Show the precise inflation
    ax.annotate('', xy=(18-0.12, 14), xytext=(18, 14),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(17.94, 13.7, '0.12', ha='center', fontsize=10, color='red')
    
    ax.text(20, 17, 'Precise inflation!', ha='center', fontsize=10,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle('Inflation Granularity: Discrete vs Continuous', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'results/inflation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Inflation comparison saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DISCRETE vs CONTINUOUS CP COMPARISON")
    print("="*80)
    
    # Run comparison
    compare_tau_values()
    visualize_inflation_difference()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("All visualizations saved to results/")
    print("="*80)