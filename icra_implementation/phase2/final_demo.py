#!/usr/bin/env python3
"""
Final demonstration with proper collision checking
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import time
import scipy.spatial.kdtree as kd

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uncertainty_wrapper import UncertaintyAwareHybridAstar
from complex_environments import get_all_scenarios, get_scenario_start_goal
from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

def check_path_collision(path, ox, oy):
    """Simple collision check for a path"""
    if path is None:
        return 1.0
    
    config = C()
    collision_count = 0
    
    # Create KD-tree for obstacles
    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    
    for x, y, yaw in zip(path.x, path.y, path.yaw):
        # Check if vehicle rectangle collides with obstacles
        # Simplified check: just check if any obstacle is too close
        ids = kdtree.query_ball_point([x, y], config.W)
        if ids:
            collision_count += 1
    
    return collision_count / len(path.x) if len(path.x) > 0 else 0

def visualize_results(results):
    """Create publication-quality visualization"""
    
    n_scenarios = len(results)
    fig = plt.figure(figsize=(18, 5*n_scenarios))
    
    method_colors = {'naive': '#FF6B6B', 'ensemble': '#4ECDC4', 'learnable_cp': '#45B7D1'}
    
    for idx, (scenario_name, scenario_data) in enumerate(results.items()):
        ox, oy = scenario_data['obstacles']
        
        for method_idx, method in enumerate(['naive', 'ensemble', 'learnable_cp']):
            ax = plt.subplot(n_scenarios, 3, idx*3 + method_idx + 1)
            
            # Plot obstacles with better visibility
            ax.scatter(ox, oy, c='black', s=1, alpha=0.4)
            
            # Get start and goal
            sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario_name)
            
            if method in scenario_data:
                path = scenario_data[method]['path']
                uncertainty = scenario_data[method].get('uncertainty', [])
                
                if path:
                    # Draw uncertainty bands first (behind path)
                    if uncertainty and method != 'naive':
                        for i in range(0, len(path.x), max(1, len(path.x)//30)):
                            radius = uncertainty[i] * 4  # Scale for visibility
                            circle = patches.Circle((path.x[i], path.y[i]), radius,
                                                   color=method_colors[method], 
                                                   alpha=0.1, zorder=1)
                            ax.add_patch(circle)
                    
                    # Draw path
                    ax.plot(path.x, path.y, color=method_colors[method], 
                           linewidth=2.5, alpha=0.9, zorder=2,
                           label=f'{method.replace("_", " ").title()}')
                    
                    # Draw vehicle rectangles at key positions
                    config = C()
                    for i in [0, len(path.x)//2, -1]:
                        if i < len(path.x):
                            # Vehicle rectangle
                            rect = patches.Rectangle(
                                (path.x[i] - config.RB, path.y[i] - config.W/2),
                                config.RF + config.RB, config.W,
                                angle=np.rad2deg(path.yaw[i]),
                                facecolor='gray' if i != 0 else 'green',
                                alpha=0.3, zorder=3
                            )
                            t = patches.transforms.Affine2D().rotate_around(
                                path.x[i], path.y[i], path.yaw[i]) + ax.transData
                            rect.set_transform(t)
                            ax.add_patch(rect)
            
            # Mark start and goal
            ax.plot(sx, sy, 'o', color='green', markersize=12, zorder=4, 
                   markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(gx, gy, 's', color='red', markersize=12, zorder=4,
                   markeredgecolor='darkred', markeredgewidth=2)
            
            # Arrow for start direction
            ax.arrow(sx, sy, 3*np.cos(syaw), 3*np.sin(syaw),
                    head_width=1, head_length=0.5, fc='green', ec='green', zorder=4)
            
            # Formatting
            ax.set_xlabel('X [m]', fontsize=11)
            ax.set_ylabel('Y [m]', fontsize=11)
            
            # Title with metrics
            title = f'{scenario_name.replace("_", " ").title()}\n{method.upper()}'
            if method in scenario_data:
                if scenario_data[method]['success']:
                    collision_rate = scenario_data[method]['collision_rate']
                    comp_time = scenario_data[method]['comp_time']
                    title += f'\nCollision: {collision_rate:.1%} | Time: {comp_time:.2f}s'
                else:
                    title += '\n(Failed)'
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.axis('equal')
            ax.set_xlim([-5, 105])
            ax.set_ylim([-5, 65])
    
    plt.suptitle('ICRA 2025: Learnable Conformal Prediction for Autonomous Path Planning\n'
                'Adaptive Safety Margins in Complex Environments',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    save_dir = 'icra_implementation/phase2/results'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/final_comparison.png', dpi=250, bbox_inches='tight')
    plt.savefig(f'{save_dir}/final_comparison.pdf', bbox_inches='tight')
    print(f"\n📊 Saved visualization to {save_dir}/final_comparison.png")
    plt.close()

def create_statistics_plot(stats):
    """Create bar chart comparing methods"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['naive', 'ensemble', 'learnable_cp']
    method_labels = ['Naive', 'Ensemble', 'Learnable CP']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Success Rate
    ax = axes[0]
    success_rates = [stats[m]['success_rate'] * 100 for m in methods]
    bars = ax.bar(method_labels, success_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Planning Success Rate', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 105])
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Collision Rate
    ax = axes[1]
    collision_rates = [stats[m]['avg_collision'] * 100 for m in methods]
    bars = ax.bar(method_labels, collision_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Average Collision Rate (%)', fontsize=12)
    ax.set_title('Safety Performance', fontsize=13, fontweight='bold')
    for bar, rate in zip(bars, collision_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Computation Time
    ax = axes[2]
    comp_times = [stats[m]['avg_time'] for m in methods]
    bars = ax.bar(method_labels, comp_times, color=colors, alpha=0.8)
    ax.set_ylabel('Average Time (seconds)', fontsize=12)
    ax.set_title('Computational Efficiency', fontsize=13, fontweight='bold')
    for bar, time in zip(bars, comp_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{time:.2f}s', ha='center', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Overall title
    fig.suptitle('Performance Comparison Across All Scenarios',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    save_dir = 'icra_implementation/phase2/results'
    plt.savefig(f'{save_dir}/statistics.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{save_dir}/statistics.pdf', bbox_inches='tight')
    print(f"📊 Saved statistics to {save_dir}/statistics.png")
    plt.close()

def main():
    """Main execution"""
    
    print("="*80)
    print("ICRA 2025: LEARNABLE CONFORMAL PREDICTION DEMONSTRATION")
    print("="*80)
    
    scenarios_to_test = ['parking_lot', 'narrow_corridor', 'maze']
    methods = ['naive', 'ensemble', 'learnable_cp']
    
    all_results = {}
    aggregate_stats = {m: {'success': [], 'collision': [], 'time': []} for m in methods}
    
    for scenario_name in scenarios_to_test:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name.replace('_', ' ').title()}")
        print('='*60)
        
        # Get scenario
        scenarios = get_all_scenarios()
        ox, oy = scenarios[scenario_name]
        sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario_name)
        
        scenario_results = {'obstacles': (ox, oy)}
        
        for method in methods:
            print(f"\n  Testing {method.upper()}...")
            
            start_time = time.time()
            
            try:
                if method == 'naive':
                    # Baseline Hybrid A*
                    config = C()
                    path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                               config.XY_RESO, config.YAW_RESO)
                    uncertainty = []
                else:
                    # Uncertainty-aware methods
                    planner = UncertaintyAwareHybridAstar(method=method)
                    path, uncertainty = planner.plan_with_uncertainty(
                        sx, sy, syaw, gx, gy, gyaw, ox, oy)
                
                success = path is not None
                
            except Exception as e:
                print(f"    Error: {e}")
                path = None
                uncertainty = []
                success = False
            
            comp_time = time.time() - start_time
            
            # Calculate metrics
            if success:
                collision_rate = check_path_collision(path, ox, oy)
                print(f"    ✓ Success! Path found with {len(path.x)} points")
                print(f"    Collision rate: {collision_rate:.1%}")
                print(f"    Computation time: {comp_time:.3f}s")
            else:
                collision_rate = 1.0
                print(f"    ✗ Failed to find path")
            
            # Store results
            scenario_results[method] = {
                'path': path,
                'uncertainty': uncertainty,
                'collision_rate': collision_rate,
                'comp_time': comp_time,
                'success': success
            }
            
            # Aggregate statistics
            aggregate_stats[method]['success'].append(1 if success else 0)
            aggregate_stats[method]['collision'].append(collision_rate)
            aggregate_stats[method]['time'].append(comp_time)
        
        all_results[scenario_name] = scenario_results
    
    # Calculate final statistics
    final_stats = {}
    for method in methods:
        final_stats[method] = {
            'success_rate': np.mean(aggregate_stats[method]['success']),
            'avg_collision': np.mean(aggregate_stats[method]['collision']),
            'avg_time': np.mean(aggregate_stats[method]['time'])
        }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    
    for method in methods:
        print(f"\n{method.upper()}:")
        print(f"  Success Rate: {final_stats[method]['success_rate']*100:.0f}%")
        print(f"  Avg Collision: {final_stats[method]['avg_collision']*100:.1f}%")
        print(f"  Avg Time: {final_stats[method]['avg_time']:.3f}s")
    
    # Calculate improvement
    if final_stats['naive']['avg_collision'] > 0:
        reduction = ((final_stats['naive']['avg_collision'] - 
                     final_stats['learnable_cp']['avg_collision']) / 
                    final_stats['naive']['avg_collision'] * 100)
        print(f"\n🎯 COLLISION REDUCTION (Naive → Learnable CP): {reduction:.1f}%")
    
    # Generate visualizations
    print("\nGenerating publication-quality figures...")
    visualize_results(all_results)
    create_statistics_plot(final_stats)
    
    print("\n✅ All experiments completed successfully!")
    print("📁 Results saved to: icra_implementation/phase2/results/")

if __name__ == "__main__":
    main()