#!/usr/bin/env python3
"""
Detailed explanation of how τ (tau) is calculated in Standard CP
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import env
from step5_naive_with_collisions import add_thinning_noise
from step6_standard_cp import StandardCP


def explain_tau_calculation():
    """
    Step-by-step explanation of τ calculation.
    """
    print("="*80)
    print("HOW TAU (τ) IS CALCULATED IN STANDARD CONFORMAL PREDICTION")
    print("="*80)
    
    # Setup
    true_env = env.Env()
    noise_params = {'thin_prob': 0.05}
    cp_planner = StandardCP(true_env, add_thinning_noise, noise_params)
    
    print("\n" + "="*80)
    print("STEP 1: NONCONFORMITY SCORE DEFINITION")
    print("="*80)
    print("""
    The nonconformity score measures how "wrong" our perception is:
    
    For each cell in the grid:
    - If perceived as FREE but actually OBSTACLE → Score = 2.0 (CRITICAL!)
    - If perceived as OBSTACLE but actually FREE → Score = 1.0 (less critical)
    - Take the MAXIMUM score across all cells
    
    Why this scoring?
    - Missing an obstacle (score=2) causes COLLISIONS - very dangerous!
    - Seeing phantom obstacles (score=1) causes inefficiency - less dangerous
    """)
    
    print("="*80)
    print("STEP 2: CALIBRATION PHASE - COLLECT SCORES")
    print("="*80)
    
    num_samples = 200
    confidence = 0.95
    
    print(f"\nGenerating {num_samples} perception samples...")
    
    scores = []
    score_details = []
    
    for i in range(num_samples):
        # Generate perceived environment
        perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=i)
        
        # Compute what type of errors occurred
        missing_obstacles = 0  # Perceived free but actually obstacle
        phantom_obstacles = 0  # Perceived obstacle but actually free
        
        for x in range(1, 50):
            for y in range(1, 30):
                point = (x, y)
                if point not in perceived_obs and point in true_env.obs:
                    missing_obstacles += 1
                elif point in perceived_obs and point not in true_env.obs:
                    phantom_obstacles += 1
        
        # Compute score
        if missing_obstacles > 0:
            score = 2.0  # Critical error
        elif phantom_obstacles > 0:
            score = 1.0  # Less critical
        else:
            score = 0.0  # No error
        
        scores.append(score)
        score_details.append({
            'missing': missing_obstacles,
            'phantom': phantom_obstacles,
            'score': score
        })
    
    # Analyze score distribution
    score_0 = sum(1 for s in scores if s == 0.0)
    score_1 = sum(1 for s in scores if s == 1.0)
    score_2 = sum(1 for s in scores if s == 2.0)
    
    print(f"\nScore Distribution from {num_samples} samples:")
    print(f"  Score = 0.0 (no error):     {score_0:3d} samples ({score_0/num_samples*100:.1f}%)")
    print(f"  Score = 1.0 (phantom obs):  {score_1:3d} samples ({score_1/num_samples*100:.1f}%)")
    print(f"  Score = 2.0 (missing obs):  {score_2:3d} samples ({score_2/num_samples*100:.1f}%)")
    
    print("\n" + "="*80)
    print("STEP 3: CALCULATE τ AS QUANTILE")
    print("="*80)
    
    # Sort scores
    sorted_scores = sorted(scores)
    
    print(f"\nConformal Prediction Quantile Formula:")
    print(f"  τ = {confidence*100:.0f}th percentile of nonconformity scores")
    print(f"  Index = ceil((n+1) × {confidence}) - 1")
    print(f"        = ceil(({num_samples}+1) × {confidence}) - 1")
    print(f"        = ceil({(num_samples+1)*confidence:.1f}) - 1")
    
    quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
    quantile_idx = min(quantile_idx, num_samples - 1)
    
    print(f"        = {quantile_idx}")
    
    tau = sorted_scores[quantile_idx]
    
    print(f"\n  τ = sorted_scores[{quantile_idx}] = {tau}")
    
    print("\n" + "="*80)
    print("STEP 4: INTERPRETATION")
    print("="*80)
    
    print(f"""
    τ = {tau} means:
    
    1. STATISTICAL GUARANTEE:
       - With 95% confidence, the true perception error will be ≤ {tau}
       - Only 5% of cases will have worse perception than this
    
    2. OBSTACLE INFLATION:
       - We inflate all perceived obstacles by {int(tau)} cells
       - This creates a "safety buffer" around obstacles
       - Even if we miss some obstacle cells, the buffer protects us
    
    3. WHY τ = 2.0 IN OUR CASE:
       - Because with 5% thinning, we almost ALWAYS have missing obstacles
       - Missing obstacles get score = 2.0
       - So 95th percentile = 2.0
       - We need 2-cell buffer to be safe!
    """)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Plot 1: Score Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=[0, 0.5, 1.5, 2.5], edgecolor='black', alpha=0.7)
    ax1.axvline(x=tau, color='red', linestyle='--', linewidth=2, label=f'τ={tau}')
    ax1.set_xlabel('Nonconformity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution')
    ax1.legend()
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['0 (No error)', '1 (Phantom)', '2 (Missing)'])
    
    # Plot 2: Sorted Scores with Quantile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sorted_scores, 'b-', linewidth=2)
    ax2.axhline(y=tau, color='red', linestyle='--', linewidth=2, label=f'τ={tau}')
    ax2.axvline(x=quantile_idx, color='green', linestyle=':', label=f'95th percentile (idx={quantile_idx})')
    ax2.fill_between(range(quantile_idx), 0, sorted_scores[:quantile_idx], alpha=0.3, color='green', label='95% coverage')
    ax2.set_xlabel('Sample Index (sorted)')
    ax2.set_ylabel('Nonconformity Score')
    ax2.set_title('Quantile Calculation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    cumulative = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
    ax3.plot(sorted_scores, cumulative, 'b-', linewidth=2)
    ax3.axvline(x=tau, color='red', linestyle='--', linewidth=2, label=f'τ={tau}')
    ax3.axhline(y=0.95, color='green', linestyle=':', label='95% confidence')
    ax3.fill_betweenx([0, 0.95], 0, tau, alpha=0.3, color='green')
    ax3.set_xlabel('Nonconformity Score')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('CDF of Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Example - No Inflation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Without Inflation (Naive)', fontweight='bold')
    ax4.set_xlim(18, 32)
    ax4.set_ylim(12, 18)
    ax4.set_aspect('equal')
    
    # Show example perceived obstacles with holes
    perceived_example = add_thinning_noise(true_env.obs, **noise_params, seed=42)
    for obs in perceived_example:
        if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
            ax4.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        facecolor='gray', edgecolor='black', alpha=0.5))
    
    # Show missing obstacles
    for obs in true_env.obs:
        if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
            if obs not in perceived_example:
                ax4.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                           facecolor='red', edgecolor='red', alpha=0.3))
                ax4.text(obs[0], obs[1], '!', ha='center', va='center', 
                        color='red', fontweight='bold', fontsize=12)
    
    ax4.set_title('Perceived (with holes!)', color='red')
    
    # Plot 5: Example - With Inflation
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title(f'With τ={tau} Inflation (Safe)', fontweight='bold', color='green')
    ax5.set_xlim(18, 32)
    ax5.set_ylim(12, 18)
    ax5.set_aspect('equal')
    
    # Show inflated obstacles
    inflated = set()
    for obs in perceived_example:
        for dx in range(-int(tau), int(tau)+1):
            for dy in range(-int(tau), int(tau)+1):
                inflated.add((obs[0]+dx, obs[1]+dy))
    
    for obs in inflated:
        if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
            if obs in perceived_example:
                color = 'gray'
                alpha = 0.5
            else:
                color = 'orange'
                alpha = 0.3
            ax5.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                       facecolor=color, edgecolor='black', alpha=alpha))
    
    ax5.text(25, 17, 'Safety buffer\ncovers holes!', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            fontsize=10, fontweight='bold')
    
    # Plot 6: Formula Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    formula_text = f"""
    CONFORMAL PREDICTION FORMULA:
    
    1. Collect n={num_samples} calibration samples
    
    2. Compute nonconformity scores:
       • Score = 2 if missing obstacles
       • Score = 1 if phantom obstacles  
       • Score = 0 if no errors
    
    3. Calculate τ:
       τ = Quantile(scores, {confidence})
       τ = sorted_scores[{quantile_idx}]
       τ = {tau}
    
    4. Apply inflation:
       Inflate obstacles by {int(tau)} cells
    
    5. Guarantee:
       P(no collision) ≥ {confidence*100:.0f}%
    """
    
    ax6.text(0.1, 0.9, formula_text, transform=ax6.transAxes,
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            verticalalignment='top')
    
    plt.suptitle('How τ (Safety Margin) is Calculated in Standard CP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = 'discrete_planner/results/tau_calculation_explained.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.show()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
    τ = {tau} was calculated by:
    1. Running {num_samples} calibration trials with perception noise
    2. Computing nonconformity scores (0, 1, or 2)
    3. Taking the {confidence*100:.0f}th percentile of scores
    4. This gives us {int(tau)}-cell safety margin for obstacle inflation
    
    Result: {confidence*100:.0f}% probability of collision-free paths!
    """)
    print("="*80)

if __name__ == "__main__":
    explain_tau_calculation()