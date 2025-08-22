#!/usr/bin/env python3
"""
Run comprehensive experiments with all three methods using existing Hybrid A*
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import torch
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_wrapper import UncertaintyAwareHybridAstar
from complex_environments import get_all_scenarios, get_scenario_start_goal
from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C, is_collision

def calculate_path_metrics(path, ox, oy, config=C()):
    """Calculate comprehensive metrics for a path"""
    if path is None:
        return {
            'success': False,
            'length': float('inf'),
            'smoothness': float('inf'),
            'clearance': 0,
            'collision_rate': 1.0,
            'computation_time': 0
        }
    
    metrics = {'success': True}
    
    # Path length
    metrics['length'] = sum(np.sqrt((path.x[i+1] - path.x[i])**2 + 
                                   (path.y[i+1] - path.y[i])**2) 
                           for i in range(len(path.x)-1))
    
    # Smoothness (cumulative heading change)
    yaw_changes = []
    for i in range(len(path.yaw)-1):
        dyaw = abs(path.yaw[i+1] - path.yaw[i])
        if dyaw > np.pi:
            dyaw = 2*np.pi - dyaw
        yaw_changes.append(dyaw)
    metrics['smoothness'] = sum(yaw_changes)
    
    # Minimum clearance to obstacles
    min_clearance = float('inf')
    collision_count = 0
    
    for x, y, yaw in zip(path.x, path.y, path.yaw):
        # Check collision
        class P:
            def __init__(self):
                self.ox = ox
                self.oy = oy
        if is_collision(x, y, yaw, P()):
            collision_count += 1
        
        # Calculate clearance
        if ox and oy:
            clearance = min(np.sqrt((x - ox_i)**2 + (y - oy_i)**2) 
                          for ox_i, oy_i in zip(ox, oy))
            min_clearance = min(min_clearance, clearance)
    
    metrics['clearance'] = min_clearance
    metrics['collision_rate'] = collision_count / len(path.x) if len(path.x) > 0 else 0
    
    return metrics

def run_single_experiment(scenario_name, method='learnable_cp', visualize=False):
    """Run a single experiment with given scenario and method"""
    
    # Get scenario
    scenarios = get_all_scenarios()
    ox, oy = scenarios[scenario_name]
    sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario_name)
    
    # Create planner
    planner = UncertaintyAwareHybridAstar(method=method)
    
    # Plan with timing
    start_time = time.time()
    
    if method in ['ensemble', 'learnable_cp']:
        path, uncertainty = planner.plan_with_uncertainty(sx, sy, syaw, gx, gy, gyaw, ox, oy)
    else:  # naive
        try:
            path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, 
                                        planner.config.XY_RESO, planner.config.YAW_RESO)
            uncertainty = []
        except:
            path = None
            uncertainty = []
    
    computation_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_path_metrics(path, ox, oy)
    metrics['computation_time'] = computation_time
    metrics['method'] = method
    metrics['scenario'] = scenario_name
    
    if visualize and path is not None:
        visualize_result(path, ox, oy, sx, sy, syaw, gx, gy, gyaw, 
                        uncertainty, method, scenario_name)
    
    return metrics, path, uncertainty

def visualize_result(path, ox, oy, sx, sy, syaw, gx, gy, gyaw, 
                     uncertainty, method, scenario_name):
    """Create publication-quality visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Path with obstacles
    ax1.plot(ox, oy, 'k.', markersize=1, alpha=0.5)
    ax1.plot(path.x, path.y, 'b-', linewidth=2, label='Planned Path')
    
    # Add uncertainty visualization if available
    if uncertainty:
        for i in range(0, len(path.x), max(1, len(path.x)//20)):
            circle = plt.Circle((path.x[i], path.y[i]), 
                               uncertainty[i] * 3,  # Scale for visibility
                               color='red', alpha=0.15)
            ax1.add_patch(circle)
    
    # Start and goal
    ax1.plot(sx, sy, 'go', markersize=12, label='Start')
    ax1.plot(gx, gy, 'ro', markersize=12, label='Goal')
    
    # Vehicle footprint at key positions
    from HybridAstarPlanner.hybrid_astar import draw_car
    for i in range(0, len(path.x), max(1, len(path.x)//10)):
        draw_car(path.x[i], path.y[i], path.yaw[i], 0, 'gray')
    
    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_title(f'{scenario_name.replace("_", " ").title()} - {method.upper()}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Uncertainty profile along path
    if uncertainty:
        ax2.plot(range(len(uncertainty)), uncertainty, 'r-', linewidth=2)
        ax2.fill_between(range(len(uncertainty)), 0, uncertainty, alpha=0.3)
        ax2.set_xlabel('Path Point Index', fontsize=12)
        ax2.set_ylabel('Uncertainty Level', fontsize=12)
        ax2.set_title('Adaptive Uncertainty Along Path', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Uncertainty\n(Naive Method)', 
                ha='center', va='center', fontsize=16, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Save figure
    save_dir = 'icra_implementation/phase2/results/figures'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{scenario_name}_{method}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{scenario_name}_{method}.pdf', bbox_inches='tight')
    plt.close()

def run_monte_carlo(n_trials=1000):
    """Run Monte Carlo simulation with all methods"""
    
    methods = ['naive', 'ensemble', 'learnable_cp']
    scenarios = list(get_all_scenarios().keys())
    
    results = {method: [] for method in methods}
    
    print(f"Running Monte Carlo simulation with {n_trials} trials...")
    
    for trial in tqdm(range(n_trials), desc="Monte Carlo Progress"):
        # Random scenario
        scenario = np.random.choice(scenarios)
        
        # Add random noise to start/goal
        sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario)
        sx += np.random.uniform(-2, 2)
        sy += np.random.uniform(-2, 2)
        gx += np.random.uniform(-2, 2)
        gy += np.random.uniform(-2, 2)
        
        # Get obstacles
        ox, oy = get_all_scenarios()[scenario]
        
        # Add some random obstacles
        n_random = np.random.randint(0, 10)
        for _ in range(n_random):
            rx = np.random.uniform(10, 90)
            ry = np.random.uniform(10, 50)
            for angle in np.linspace(0, 2*np.pi, 8):
                ox.append(rx + 2*np.cos(angle))
                oy.append(ry + 2*np.sin(angle))
        
        # Test each method
        for method in methods:
            try:
                planner = UncertaintyAwareHybridAstar(method=method)
                
                if method in ['ensemble', 'learnable_cp']:
                    path, uncertainty = planner.plan_with_uncertainty(
                        sx, sy, syaw, gx, gy, gyaw, ox, oy)
                else:
                    path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                                planner.config.XY_RESO, planner.config.YAW_RESO)
                
                metrics = calculate_path_metrics(path, ox, oy)
                metrics['trial'] = trial
                metrics['scenario'] = scenario
                results[method].append(metrics)
                
            except Exception as e:
                # Planning failed
                results[method].append({
                    'success': False,
                    'length': float('inf'),
                    'smoothness': float('inf'),
                    'clearance': 0,
                    'collision_rate': 1.0,
                    'computation_time': 0,
                    'trial': trial,
                    'scenario': scenario
                })
    
    return results

def analyze_results(results):
    """Analyze Monte Carlo results and generate statistics"""
    
    stats = {}
    
    for method, trials in results.items():
        successful = [t for t in trials if t['success']]
        
        stats[method] = {
            'success_rate': len(successful) / len(trials),
            'avg_length': np.mean([t['length'] for t in successful]) if successful else float('inf'),
            'avg_smoothness': np.mean([t['smoothness'] for t in successful]) if successful else float('inf'),
            'avg_clearance': np.mean([t['clearance'] for t in successful]) if successful else 0,
            'collision_rate': np.mean([t['collision_rate'] for t in trials]),
            'avg_computation_time': np.mean([t['computation_time'] for t in trials]),
            'std_length': np.std([t['length'] for t in successful]) if successful else 0,
            'std_clearance': np.std([t['clearance'] for t in successful]) if successful else 0,
        }
    
    # Statistical tests
    from scipy import stats as scipy_stats
    
    # Compare collision rates
    naive_collisions = [t['collision_rate'] for t in results['naive']]
    cp_collisions = [t['collision_rate'] for t in results['learnable_cp']]
    
    if naive_collisions and cp_collisions:
        t_stat, p_value = scipy_stats.ttest_ind(naive_collisions, cp_collisions)
        stats['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(naive_collisions)**2 + np.std(cp_collisions)**2) / 2)
        if pooled_std > 0:
            cohens_d = (np.mean(naive_collisions) - np.mean(cp_collisions)) / pooled_std
            stats['effect_size'] = cohens_d
    
    return stats

def create_comparison_plots(results, stats):
    """Create publication-quality comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = list(results.keys())
    colors = {'naive': '#FF6B6B', 'ensemble': '#4ECDC4', 'learnable_cp': '#45B7D1'}
    
    # 1. Success Rate
    ax = axes[0, 0]
    success_rates = [stats[m]['success_rate'] * 100 for m in methods]
    bars = ax.bar(methods, success_rates, color=[colors[m] for m in methods])
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Planning Success Rate', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=10)
    
    # 2. Collision Rate
    ax = axes[0, 1]
    collision_rates = [stats[m]['collision_rate'] * 100 for m in methods]
    bars = ax.bar(methods, collision_rates, color=[colors[m] for m in methods])
    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.set_title('Path Collision Rate', fontsize=14, fontweight='bold')
    for bar, rate in zip(bars, collision_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=10)
    
    # 3. Average Clearance
    ax = axes[0, 2]
    clearances = [stats[m]['avg_clearance'] for m in methods]
    bars = ax.bar(methods, clearances, color=[colors[m] for m in methods])
    ax.set_ylabel('Average Clearance (m)', fontsize=12)
    ax.set_title('Obstacle Clearance', fontsize=14, fontweight='bold')
    for bar, clear in zip(bars, clearances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{clear:.2f}', ha='center', fontsize=10)
    
    # 4. Path Length Distribution
    ax = axes[1, 0]
    for method in methods:
        lengths = [t['length'] for t in results[method] if t['success'] and t['length'] < 200]
        if lengths:
            ax.hist(lengths, bins=30, alpha=0.6, label=method, color=colors[method])
    ax.set_xlabel('Path Length (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Path Length Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 5. Computation Time
    ax = axes[1, 1]
    comp_times = [stats[m]['avg_computation_time'] for m in methods]
    bars = ax.bar(methods, comp_times, color=[colors[m] for m in methods])
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Average Planning Time', fontsize=14, fontweight='bold')
    for bar, time in zip(bars, comp_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.3f}s', ha='center', fontsize=10)
    
    # 6. Statistical Significance
    ax = axes[1, 2]
    if 'statistical_test' in stats:
        text = f"Statistical Comparison\n(Naive vs Learnable CP)\n\n"
        text += f"t-statistic: {stats['statistical_test']['t_statistic']:.3f}\n"
        text += f"p-value: {stats['statistical_test']['p_value']:.2e}\n"
        if 'effect_size' in stats:
            text += f"Cohen's d: {stats['effect_size']:.3f}\n"
        text += f"\nResult: {'SIGNIFICANT' if stats['statistical_test']['significant'] else 'Not Significant'}"
        
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Statistical Analysis', fontsize=14, fontweight='bold')
    
    plt.suptitle('ICRA 2025: Learnable Conformal Prediction for Path Planning',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    save_dir = 'icra_implementation/phase2/results'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/comparison_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/comparison_results.pdf', bbox_inches='tight')
    plt.close()

def main():
    """Main execution function"""
    
    # Create results directory
    os.makedirs('icra_implementation/phase2/results', exist_ok=True)
    
    print("=" * 60)
    print("ICRA 2025: Comprehensive Experiments")
    print("=" * 60)
    
    # 1. Test individual scenarios with visualization
    print("\n1. Testing individual scenarios...")
    scenarios = ['parking_lot', 'narrow_corridor', 'maze', 'roundabout']
    methods = ['naive', 'ensemble', 'learnable_cp']
    
    individual_results = []
    for scenario in scenarios:
        print(f"\n  Testing {scenario}...")
        for method in methods:
            print(f"    Method: {method}")
            metrics, path, uncertainty = run_single_experiment(
                scenario, method, visualize=True)
            individual_results.append(metrics)
            print(f"      Success: {metrics['success']}, "
                  f"Collision Rate: {metrics['collision_rate']:.2%}")
    
    # 2. Run Monte Carlo simulation
    print("\n2. Running Monte Carlo simulation...")
    monte_carlo_results = run_monte_carlo(n_trials=1000)
    
    # 3. Analyze results
    print("\n3. Analyzing results...")
    stats = analyze_results(monte_carlo_results)
    
    # 4. Create comparison plots
    print("\n4. Generating publication-quality figures...")
    create_comparison_plots(monte_carlo_results, stats)
    
    # 5. Save results
    print("\n5. Saving results...")
    
    # Save statistics
    with open('icra_implementation/phase2/results/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save raw results
    np.save('icra_implementation/phase2/results/monte_carlo_results.npy', monte_carlo_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    for method in methods:
        print(f"\n{method.upper()}:")
        print(f"  Success Rate: {stats[method]['success_rate']:.1%}")
        print(f"  Collision Rate: {stats[method]['collision_rate']:.2%}")
        print(f"  Avg Clearance: {stats[method]['avg_clearance']:.2f}m")
        print(f"  Avg Path Length: {stats[method]['avg_length']:.1f}m")
        print(f"  Avg Computation: {stats[method]['avg_computation_time']:.3f}s")
    
    if 'statistical_test' in stats:
        print(f"\nStatistical Significance:")
        print(f"  p-value: {stats['statistical_test']['p_value']:.2e}")
        print(f"  Significant: {stats['statistical_test']['significant']}")
        if 'effect_size' in stats:
            print(f"  Effect Size (Cohen's d): {stats['effect_size']:.3f}")
    
    # Calculate improvement
    naive_collision = stats['naive']['collision_rate']
    cp_collision = stats['learnable_cp']['collision_rate']
    if naive_collision > 0:
        improvement = (naive_collision - cp_collision) / naive_collision * 100
        print(f"\n🎯 Collision Reduction: {improvement:.1f}%")
    
    print("\n✅ All experiments completed successfully!")
    print(f"📊 Results saved to: icra_implementation/phase2/results/")

if __name__ == "__main__":
    main()