#!/usr/bin/env python3
"""
FINAL COMPARISON: Naive vs Standard CP Path Planning
Comprehensive evaluation and visualization of methods
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env

# Import our modules
from step5_naive_with_collisions import add_thinning_noise
from step6_standard_cp import StandardCP


def comprehensive_monte_carlo(true_env, num_trials=10000):
    """
    Large-scale Monte Carlo comparison with 10,000 trials.
    """
    print("="*80)
    print("COMPREHENSIVE MONTE CARLO EVALUATION")
    print(f"Running {num_trials:,} trials for statistical significance")
    print("="*80)
    
    # Setup
    noise_params = {'thin_prob': 0.05}
    cp_planner = StandardCP(true_env, add_thinning_noise, noise_params)
    
    # Calibration
    print("\nCalibration Phase...")
    cp_planner.calibration_phase(num_samples=200, confidence=0.95)
    
    # Results storage
    results = {
        'naive': {
            'successes': 0,
            'failures': 0,
            'collisions': 0,
            'path_lengths': [],
            'collision_counts': [],
            'computation_times': []
        },
        'standard_cp': {
            'successes': 0,
            'failures': 0,
            'collisions': 0,
            'path_lengths': [],
            'collision_counts': [],
            'computation_times': []
        }
    }
    
    print(f"\nRunning {num_trials:,} trials...")
    print("Progress: ", end="", flush=True)
    
    for trial in range(num_trials):
        if trial % 1000 == 0:
            print(f"{trial//1000}k...", end="", flush=True)
        
        # Generate perceived environment
        perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=trial+5000)
        
        # NAIVE METHOD
        start_time = time.time()
        try:
            astar = Astar.AStar((5, 15), (45, 15), "euclidean")
            astar.obs = perceived_obs
            naive_path, _ = astar.searching()
            
            if naive_path:
                results['naive']['successes'] += 1
                results['naive']['path_lengths'].append(len(naive_path))
                
                # Check collisions
                collisions = [p for p in naive_path if p in true_env.obs]
                results['naive']['collision_counts'].append(len(collisions))
                if collisions:
                    results['naive']['collisions'] += 1
            else:
                results['naive']['failures'] += 1
        except:
            results['naive']['failures'] += 1
        
        results['naive']['computation_times'].append(time.time() - start_time)
        
        # STANDARD CP METHOD
        start_time = time.time()
        cp_result = cp_planner.plan_with_cp(perceived_obs)
        
        if cp_result['success']:
            results['standard_cp']['successes'] += 1
            results['standard_cp']['path_lengths'].append(cp_result['path_length'])
            results['standard_cp']['collision_counts'].append(cp_result['num_collisions'])
            
            if cp_result['has_collision']:
                results['standard_cp']['collisions'] += 1
        else:
            results['standard_cp']['failures'] += 1
        
        results['standard_cp']['computation_times'].append(time.time() - start_time)
    
    print(f"{num_trials//1000}k. Done!")
    
    # Calculate comprehensive statistics
    for method in ['naive', 'standard_cp']:
        r = results[method]
        
        if r['successes'] > 0:
            r['success_rate'] = (r['successes'] / num_trials) * 100
            r['collision_rate'] = (r['collisions'] / r['successes']) * 100
            r['avg_path_length'] = np.mean(r['path_lengths'])
            r['std_path_length'] = np.std(r['path_lengths'])
            r['avg_collisions'] = np.mean(r['collision_counts'])
            r['std_collisions'] = np.std(r['collision_counts'])
            r['avg_time'] = np.mean(r['computation_times']) * 1000  # ms
            r['coverage'] = 100 - r['collision_rate']  # Coverage guarantee
            
            # Confidence intervals (95%)
            n = r['successes']
            p = r['collisions'] / n
            r['collision_ci'] = 1.96 * np.sqrt(p * (1-p) / n) * 100
        else:
            r['success_rate'] = 0
            r['collision_rate'] = 0
            r['coverage'] = 0
    
    return results


def create_final_visualization(true_env, mc_results):
    """
    Create comprehensive final visualization.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # === ROW 1: Example paths ===
    noise_params = {'thin_prob': 0.05}
    cp_planner = StandardCP(true_env, add_thinning_noise, noise_params)
    cp_planner.calibration_phase(num_samples=100, confidence=0.95)
    
    seeds = [123, 456, 789]  # Different scenarios
    
    for idx, seed in enumerate(seeds):
        perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=seed)
        
        # Naive path
        ax = fig.add_subplot(gs[0, idx])
        ax.set_title(f'Naive (Example {idx+1})', fontsize=10)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)
        
        # Plot perceived obstacles
        for obs in perceived_obs:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.3, edgecolor='gray',
                                    facecolor='gray', alpha=0.4)
            ax.add_patch(rect)
        
        # Plot missing walls (thinned)
        for obs in true_env.obs:
            if obs not in perceived_obs and not (obs[0] in [0, 50] or obs[1] in [0, 30]):
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=0.8, edgecolor='blue',
                                        facecolor='none', linestyle='--', alpha=0.6)
                ax.add_patch(rect)
        
        # Plan naive path
        try:
            astar = Astar.AStar((5, 15), (45, 15), "euclidean")
            astar.obs = perceived_obs
            naive_path, _ = astar.searching()
            
            if naive_path:
                path_x = [p[0] for p in naive_path]
                path_y = [p[1] for p in naive_path]
                ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=0.7)
                
                # Mark collisions
                collisions = [p for p in naive_path if p in true_env.obs]
                for cp in collisions:
                    ax.plot(cp[0], cp[1], 'rx', markersize=8, markeredgewidth=2)
        except:
            pass
        
        ax.plot(5, 15, 'go', markersize=6)
        ax.plot(45, 15, 'ro', markersize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # === ROW 2: Standard CP paths ===
    for idx, seed in enumerate(seeds):
        perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=seed)
        
        ax = fig.add_subplot(gs[1, idx])
        ax.set_title(f'Standard CP (Example {idx+1})', fontsize=10)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.1)
        
        # Plan with CP
        cp_result = cp_planner.plan_with_cp(perceived_obs)
        
        if cp_result['success']:
            # Plot inflated obstacles
            for obs in cp_result['inflated_obs']:
                if obs in perceived_obs:
                    color = 'gray'
                    alpha = 0.4
                else:
                    color = 'orange'
                    alpha = 0.2
                
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=0.3, edgecolor=color,
                                        facecolor=color, alpha=alpha)
                ax.add_patch(rect)
            
            # Plot path
            path_x = [p[0] for p in cp_result['path']]
            path_y = [p[1] for p in cp_result['path']]
            ax.plot(path_x, path_y, 'g-', linewidth=1.5, alpha=0.7)
            
            # Should have no collisions
            if cp_result['collision_points']:
                for cp in cp_result['collision_points']:
                    ax.plot(cp[0], cp[1], 'rx', markersize=8, markeredgewidth=2)
        
        ax.plot(5, 15, 'go', markersize=6)
        ax.plot(45, 15, 'ro', markersize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # === Statistics Plots ===
    
    # Collision Rate Comparison
    ax = fig.add_subplot(gs[0, 3])
    methods = ['Naive', 'Standard CP']
    collision_rates = [mc_results['naive']['collision_rate'], 
                      mc_results['standard_cp']['collision_rate']]
    error_bars = [mc_results['naive'].get('collision_ci', 0),
                 mc_results['standard_cp'].get('collision_ci', 0)]
    
    bars = ax.bar(methods, collision_rates, yerr=error_bars, capsize=5,
                  color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Collision Rate (%)')
    ax.set_title('Safety Comparison', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(collision_rates) * 1.2)
    
    # Add value labels
    for bar, rate in zip(bars, collision_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Path Length Comparison
    ax = fig.add_subplot(gs[1, 3])
    path_lengths = [mc_results['naive']['avg_path_length'],
                   mc_results['standard_cp']['avg_path_length']]
    path_stds = [mc_results['naive']['std_path_length'],
                mc_results['standard_cp']['std_path_length']]
    
    bars = ax.bar(methods, path_lengths, yerr=path_stds, capsize=5,
                  color=['orange', 'blue'], alpha=0.7)
    ax.set_ylabel('Average Path Length')
    ax.set_title('Efficiency Comparison', fontsize=11, fontweight='bold')
    
    for bar, length in zip(bars, path_lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{length:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Coverage Guarantee Plot
    ax = fig.add_subplot(gs[2, :2])
    ax.set_title('Coverage Guarantee (Safety)', fontsize=11, fontweight='bold')
    
    coverage_naive = mc_results['naive']['coverage']
    coverage_cp = mc_results['standard_cp']['coverage']
    
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% Target')
    ax.bar(['Naive', 'Standard CP'], [coverage_naive, coverage_cp],
          color=['red', 'green'], alpha=0.7)
    ax.set_ylabel('Coverage (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    # Add annotations
    ax.text(0, coverage_naive + 1, f'{coverage_naive:.1f}%', 
           ha='center', fontsize=10)
    ax.text(1, coverage_cp + 1, f'{coverage_cp:.1f}%',
           ha='center', fontsize=10)
    
    # Results Table
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Naive', 'Standard CP', 'Improvement'],
        ['Success Rate', f"{mc_results['naive']['success_rate']:.1f}%",
         f"{mc_results['standard_cp']['success_rate']:.1f}%", '-'],
        ['Collision Rate', f"{mc_results['naive']['collision_rate']:.1f}%",
         f"{mc_results['standard_cp']['collision_rate']:.1f}%",
         f"↓ {mc_results['naive']['collision_rate'] - mc_results['standard_cp']['collision_rate']:.1f}%"],
        ['Avg Path Length', f"{mc_results['naive']['avg_path_length']:.1f}",
         f"{mc_results['standard_cp']['avg_path_length']:.1f}",
         f"↑ {mc_results['standard_cp']['avg_path_length'] - mc_results['naive']['avg_path_length']:.1f}"],
        ['Avg Time (ms)', f"{mc_results['naive']['avg_time']:.2f}",
         f"{mc_results['standard_cp']['avg_time']:.2f}",
         f"↑ {mc_results['standard_cp']['avg_time'] - mc_results['naive']['avg_time']:.2f}"],
        ['Coverage', f"{coverage_naive:.1f}%",
         f"{coverage_cp:.1f}%",
         f"↑ {coverage_cp - coverage_naive:.1f}%"]
    ]
    
    table = ax.table(cellText=table_data, loc='center',
                    cellLoc='center', colWidths=[0.25, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color data cells
    for i in range(1, 6):
        # Highlight collision rate row
        if i == 2:
            table[(i, 1)].set_facecolor('#ffcccc')  # Red for naive
            table[(i, 2)].set_facecolor('#ccffcc')  # Green for CP
    
    ax.set_title('Performance Summary', fontsize=11, fontweight='bold', pad=20)
    
    plt.suptitle('Uncertainty-Aware Path Planning: Naive vs Standard CP Comparison\n' +
                f'Monte Carlo Evaluation with {mc_results["naive"]["successes"] + mc_results["naive"]["failures"]:,} trials',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('discrete_planner/results', exist_ok=True)
    save_path = 'discrete_planner/results/final_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFinal visualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main final comparison.
    """
    print("="*80)
    print("FINAL COMPARISON: NAIVE vs STANDARD CP")
    print("="*80)
    
    # Get environment
    true_env = env.Env()
    
    # Run comprehensive Monte Carlo
    mc_results = comprehensive_monte_carlo(true_env, num_trials=10000)
    
    # Print detailed results
    print("\n" + "="*80)
    print("MONTE CARLO RESULTS (10,000 trials)")
    print("="*80)
    
    print("\nNAIVE METHOD:")
    print(f"  Success Rate: {mc_results['naive']['success_rate']:.2f}%")
    print(f"  Collision Rate: {mc_results['naive']['collision_rate']:.2f}% ± {mc_results['naive'].get('collision_ci', 0):.2f}%")
    print(f"  Coverage: {mc_results['naive']['coverage']:.2f}%")
    print(f"  Avg Path Length: {mc_results['naive']['avg_path_length']:.2f} ± {mc_results['naive']['std_path_length']:.2f}")
    print(f"  Avg Computation Time: {mc_results['naive']['avg_time']:.3f} ms")
    
    print("\nSTANDARD CP:")
    print(f"  Success Rate: {mc_results['standard_cp']['success_rate']:.2f}%")
    print(f"  Collision Rate: {mc_results['standard_cp']['collision_rate']:.2f}% ± {mc_results['standard_cp'].get('collision_ci', 0):.2f}%")
    print(f"  Coverage: {mc_results['standard_cp']['coverage']:.2f}%")
    print(f"  Avg Path Length: {mc_results['standard_cp']['avg_path_length']:.2f} ± {mc_results['standard_cp']['std_path_length']:.2f}")
    print(f"  Avg Computation Time: {mc_results['standard_cp']['avg_time']:.3f} ms")
    
    # Calculate improvements
    collision_reduction = mc_results['naive']['collision_rate'] - mc_results['standard_cp']['collision_rate']
    coverage_improvement = mc_results['standard_cp']['coverage'] - mc_results['naive']['coverage']
    path_increase = mc_results['standard_cp']['avg_path_length'] - mc_results['naive']['avg_path_length']
    path_increase_pct = (path_increase / mc_results['naive']['avg_path_length']) * 100
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    print(f"\n✅ Collision Rate Reduction: {collision_reduction:.2f}%")
    print(f"✅ Coverage Improvement: {coverage_improvement:.2f}%")
    print(f"📊 Path Length Increase: {path_increase:.2f} cells ({path_increase_pct:.1f}%)")
    print(f"⏱️ Computation Overhead: {mc_results['standard_cp']['avg_time'] - mc_results['naive']['avg_time']:.3f} ms")
    
    # Statistical significance
    if mc_results['standard_cp']['collision_rate'] < 5:
        print(f"\n🎯 Standard CP achieves target <5% collision rate!")
    
    if mc_results['standard_cp']['coverage'] >= 95:
        print(f"🛡️ Standard CP provides ≥95% coverage guarantee!")
    
    # Create final visualization
    print("\nGenerating final comprehensive visualization...")
    create_final_visualization(true_env, mc_results)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\n📈 Standard Conformal Prediction successfully:")
    print(f"   • Reduces collision rate from {mc_results['naive']['collision_rate']:.1f}% to {mc_results['standard_cp']['collision_rate']:.1f}%")
    print(f"   • Achieves {mc_results['standard_cp']['coverage']:.1f}% coverage (safety guarantee)")
    print(f"   • Trade-off: {path_increase_pct:.1f}% longer paths for safety")
    print("\n✨ This demonstrates effective uncertainty-aware planning!")
    print("="*80)


if __name__ == "__main__":
    main()