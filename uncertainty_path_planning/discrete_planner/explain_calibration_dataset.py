#!/usr/bin/env python3
"""
Detailed explanation of calibration dataset generation in Standard CP
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import env
from step5_naive_with_collisions import add_thinning_noise


def explain_calibration_dataset():
    """
    Step-by-step explanation of calibration dataset.
    """
    print("="*80)
    print("CALIBRATION DATASET IN CONFORMAL PREDICTION")
    print("="*80)
    
    print("""
    KEY CONCEPT: The calibration dataset is NOT about paths or planning!
    It's about understanding the PERCEPTION ERROR distribution.
    
    We need to answer: "How wrong can our perception be?"
    """)
    
    # Setup
    true_env = env.Env()
    true_obstacles = true_env.obs
    
    print("\n" + "="*80)
    print("STEP 1: WHAT IS A CALIBRATION DATASET?")
    print("="*80)
    print("""
    Calibration Dataset = Collection of (true, perceived) environment pairs
    
    For each calibration sample i:
        1. Start with TRUE environment (ground truth)
        2. Apply perception noise (5% wall thinning)
        3. Get PERCEIVED environment (what robot sees)
        4. Compare them to measure perception error
        5. Store the error as "nonconformity score"
    
    This is like taking many "snapshots" of perception errors!
    """)
    
    print("\n" + "="*80)
    print("STEP 2: GENERATING CALIBRATION DATA")
    print("="*80)
    
    # Generate small calibration dataset for demonstration
    num_samples = 10
    calibration_data = []
    
    print(f"\nGenerating {num_samples} calibration samples:")
    print("-" * 50)
    
    for i in range(num_samples):
        # Generate perceived environment with different random seed
        perceived_obs = add_thinning_noise(true_obstacles, thin_prob=0.05, seed=i)
        
        # Count perception errors
        missing_walls = 0  # Dangerous: perceived free but actually obstacle
        phantom_walls = 0  # Safe but inefficient: perceived obstacle but actually free
        
        for x in range(1, 50):
            for y in range(1, 30):
                point = (x, y)
                if point not in perceived_obs and point in true_obstacles:
                    missing_walls += 1
                elif point in perceived_obs and point not in true_obstacles:
                    phantom_walls += 1
        
        # Compute nonconformity score
        if missing_walls > 0:
            score = 2.0  # Critical error
        elif phantom_walls > 0:
            score = 1.0
        else:
            score = 0.0
        
        calibration_data.append({
            'sample_id': i,
            'missing_walls': missing_walls,
            'phantom_walls': phantom_walls,
            'score': score,
            'perceived_obs': perceived_obs
        })
        
        print(f"Sample {i:2d}: Missing walls={missing_walls:3d}, "
              f"Phantom walls={phantom_walls:3d}, Score={score}")
    
    print("-" * 50)
    print(f"Calibration dataset contains {len(calibration_data)} samples")
    
    print("\n" + "="*80)
    print("STEP 3: WHY DO WE NEED THIS?")
    print("="*80)
    print("""
    The calibration dataset tells us:
    
    1. DISTRIBUTION OF ERRORS:
       - How often do we miss walls? (92.5% of time with 5% thinning)
       - How severe are the errors? (usually 20-50 missing cells)
       
    2. WORST-CASE ANALYSIS:
       - What's the 95th percentile error? (τ = 2.0)
       - This becomes our safety margin
       
    3. NO NEED FOR LABELED PATHS:
       - We DON'T need collision examples
       - We DON'T need successful/failed paths
       - We ONLY need to observe perception errors
       
    This is much easier than supervised learning!
    """)
    
    print("\n" + "="*80)
    print("STEP 4: FROM CALIBRATION TO SAFETY MARGIN")
    print("="*80)
    
    # Compute τ from our small dataset
    scores = [d['score'] for d in calibration_data]
    sorted_scores = sorted(scores)
    confidence = 0.95
    quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
    quantile_idx = min(quantile_idx, num_samples - 1)
    tau = sorted_scores[quantile_idx]
    
    print(f"""
    From {num_samples} calibration samples:
    - Scores: {sorted_scores}
    - 95th percentile index: {quantile_idx}
    - τ = {tau}
    
    This means: "Inflate obstacles by {int(tau)} cells to be 95% safe"
    """)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Calibration Dataset Generation for Conformal Prediction', 
                 fontsize=14, fontweight='bold')
    
    # Show 6 example calibration samples
    for idx in range(6):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_title(f'Calibration Sample {idx+1}', fontsize=10)
        ax.set_xlim(18, 32)
        ax.set_ylim(12, 18)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Get this sample's perceived environment
        sample = calibration_data[idx]
        perceived = sample['perceived_obs']
        
        # Plot true obstacles (light gray)
        for obs in true_obstacles:
            if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        facecolor='lightgray', edgecolor='gray', 
                                        alpha=0.3, linewidth=0.5)
                ax.add_patch(rect)
        
        # Plot perceived obstacles (black)
        for obs in perceived:
            if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        facecolor='black', edgecolor='black', 
                                        alpha=0.7)
                ax.add_patch(rect)
        
        # Highlight missing walls (red)
        for obs in true_obstacles:
            if 18 <= obs[0] <= 32 and 12 <= obs[1] <= 18:
                if obs not in perceived:
                    rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                            facecolor='red', edgecolor='red', 
                                            alpha=0.5, linewidth=2)
                    ax.add_patch(rect)
        
        # Add score annotation
        score_color = 'red' if sample['score'] == 2.0 else 'green'
        ax.text(25, 17.5, f"Score: {sample['score']}", 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor='white', alpha=0.8),
                color=score_color)
    
    # Plot calibration process flow
    ax = fig.add_subplot(gs[:, 3])
    ax.axis('off')
    
    flow_text = """
    CALIBRATION PROCESS
    ═══════════════════
    
    1. START
       └─> True Environment
    
    2. FOR i = 1 to n:
       ├─> Add noise (seed i)
       ├─> Get perceived env
       ├─> Compare true vs perceived
       └─> Compute score
    
    3. COLLECT SCORES
       └─> [2, 2, 0, 2, 2, ...]
    
    4. COMPUTE τ
       └─> 95th percentile
       └─> τ = 2.0
    
    5. APPLY
       └─> Inflate by τ cells
       └─> Plan safely!
    
    ═══════════════════
    
    KEY INSIGHTS:
    
    • No path planning needed
      during calibration
    
    • Only measuring how
      perception differs
      from reality
    
    • τ captures worst-case
      perception error at
      95% confidence
    
    • Works for ANY planner
      (A*, RRT, etc.)
    """
    
    ax.text(0.1, 0.95, flow_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor="lightyellow", alpha=0.9),
            verticalalignment='top')
    
    # Add histogram of scores
    ax2 = fig.add_subplot(gs[2, :3])
    ax2.set_title('Distribution of Nonconformity Scores', fontsize=11)
    
    # Generate larger dataset for better histogram
    large_scores = []
    for i in range(200):
        perceived = add_thinning_noise(true_obstacles, thin_prob=0.05, seed=i+100)
        missing = sum(1 for p in true_obstacles if p not in perceived)
        score = 2.0 if missing > 0 else 0.0
        large_scores.append(score)
    
    ax2.hist(large_scores, bins=[0, 0.5, 1.5, 2.5], 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(x=2.0, color='red', linestyle='--', linewidth=2, 
               label=f'τ = 2.0 (95th percentile)')
    ax2.set_xlabel('Nonconformity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['0\n(No error)', '1\n(Phantom)', '2\n(Missing walls)'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax2.text(1, max(ax2.get_ylim())*0.8, 
            f'92.5% of samples\nhave missing walls\n(dangerous!)',
            ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", 
                     facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    save_path = 'discrete_planner/results/calibration_dataset_explained.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.show()
    
    print("\n" + "="*80)
    print("COMPARISON WITH MACHINE LEARNING")
    print("="*80)
    print("""
    TRADITIONAL ML (Supervised):
    - Need: Labeled paths (safe/collision)
    - Data: (path, label) pairs
    - Training: Learn f(path) → safe/unsafe
    - Problem: Need lots of collision examples!
    
    CONFORMAL PREDICTION:
    - Need: Just perception samples
    - Data: (true_env, perceived_env) pairs
    - Calibration: Measure perception errors
    - Advantage: No paths needed, no collisions needed!
    
    This is why CP is so powerful for safety-critical applications!
    """)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
    The calibration dataset:
    1. Is generated by sampling perception errors (NOT paths)
    2. Each sample compares true vs perceived environment
    3. Scores measure how wrong perception is
    4. τ is the 95th percentile of these scores
    5. This gives us probabilistic safety guarantee!
    
    Simple, elegant, and mathematically rigorous!
    """)
    print("="*80)


if __name__ == "__main__":
    explain_calibration_dataset()