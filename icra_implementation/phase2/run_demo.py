#!/usr/bin/env python3
"""
Demonstration of uncertainty-aware path planning with publication-quality visualizations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
import time

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uncertainty_wrapper import UncertaintyAwareHybridAstar
from complex_environments import get_all_scenarios, get_scenario_start_goal
from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

def visualize_comparison(results, save_path='icra_implementation/phase2/results'):
    """Create publication-quality comparison visualization"""
    
    n_scenarios = len(results)
    fig = plt.figure(figsize=(20, 4*n_scenarios))
    
    for idx, (scenario_name, scenario_results) in enumerate(results.items()):
        ox, oy = scenario_results['obstacles']
        
        for method_idx, method in enumerate(['naive', 'ensemble', 'learnable_cp']):
            ax = plt.subplot(n_scenarios, 3, idx*3 + method_idx + 1)
            
            # Plot obstacles
            ax.plot(ox, oy, 'k.', markersize=0.5, alpha=0.3)
            
            # Plot path
            if method in scenario_results:
                path = scenario_results[method]['path']
                uncertainty = scenario_results[method].get('uncertainty', [])
                
                if path:
                    # Path line
                    ax.plot(path.x, path.y, 'b-', linewidth=2, alpha=0.8)
                    
                    # Uncertainty visualization for CP methods
                    if uncertainty and method in ['ensemble', 'learnable_cp']:
                        # Sample points for uncertainty display
                        n_samples = min(30, len(path.x))
                        indices = np.linspace(0, len(path.x)-1, n_samples, dtype=int)
                        
                        for i in indices:
                            # Scale uncertainty for visibility
                            radius = uncertainty[i] * 5 if method == 'learnable_cp' else uncertainty[i] * 3
                            circle = plt.Circle((path.x[i], path.y[i]), radius,
                                              color='red', alpha=0.1)
                            ax.add_patch(circle)
                    
                    # Show vehicle at start and goal
                    from HybridAstarPlanner.hybrid_astar import draw_car
                    
                    # Temporarily override matplotlib's current axes
                    original_axes = plt.gca()
                    plt.sca(ax)
                    draw_car(path.x[0], path.y[0], path.yaw[0], 0, 'green')
                    draw_car(path.x[-1], path.y[-1], path.yaw[-1], 0, 'red')
                    plt.sca(original_axes)
            
            # Start and goal markers
            sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario_name)
            ax.plot(sx, sy, 'go', markersize=8, alpha=0.7)
            ax.plot(gx, gy, 'ro', markersize=8, alpha=0.7)
            
            # Formatting
            ax.set_xlabel('X [m]' if idx == n_scenarios-1 else '')
            ax.set_ylabel('Y [m]' if method_idx == 0 else '')
            
            title = f'{scenario_name.replace("_", " ").title()}\n{method.upper()}'
            if method in scenario_results and scenario_results[method]['path']:
                collision_rate = scenario_results[method].get('collision_rate', 0)
                title += f'\n(Collision: {collision_rate:.1%})'
            ax.set_title(title, fontsize=10)
            
            ax.grid(True, alpha=0.2)
            ax.axis('equal')
            ax.set_xlim([-5, 105])
            ax.set_ylim([-5, 65])
    
    plt.suptitle('ICRA 2025: Learnable Conformal Prediction for Safe Path Planning',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/method_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{save_path}/method_comparison.pdf', bbox_inches='tight')
    print(f"Saved visualization to {save_path}/method_comparison.png")
    plt.close()

def run_scenario_comparison():
    """Run comparison across different scenarios and methods"""
    
    scenarios_to_test = ['parking_lot', 'narrow_corridor', 'maze']
    methods = ['naive', 'ensemble', 'learnable_cp']
    
    results = {}
    
    for scenario_name in scenarios_to_test:
        print(f"\n{'='*60}")
        print(f"Testing Scenario: {scenario_name.replace('_', ' ').title()}")
        print('='*60)
        
        # Get scenario
        scenarios = get_all_scenarios()
        ox, oy = scenarios[scenario_name]
        sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal(scenario_name)
        
        scenario_results = {'obstacles': (ox, oy)}
        
        for method in methods:
            print(f"\n  Method: {method.upper()}")
            
            start_time = time.time()
            
            if method == 'naive':
                # Baseline Hybrid A*
                try:
                    config = C()
                    path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                               config.XY_RESO, config.YAW_RESO)
                    uncertainty = []
                except:
                    path = None
                    uncertainty = []
            else:
                # Uncertainty-aware methods
                planner = UncertaintyAwareHybridAstar(method=method)
                try:
                    path, uncertainty = planner.plan_with_uncertainty(
                        sx, sy, syaw, gx, gy, gyaw, ox, oy)
                except:
                    path = None
                    uncertainty = []
            
            elapsed = time.time() - start_time
            
            # Calculate collision rate
            collision_rate = 0
            if path:
                from HybridAstarPlanner.hybrid_astar import is_collision
                
                class P:
                    def __init__(self):
                        self.ox = ox
                        self.oy = oy
                
                collision_count = sum(1 for x, y, yaw in zip(path.x, path.y, path.yaw)
                                     if is_collision(x, y, yaw, P()))
                collision_rate = collision_count / len(path.x) if len(path.x) > 0 else 0
            
            scenario_results[method] = {
                'path': path,
                'uncertainty': uncertainty,
                'collision_rate': collision_rate,
                'computation_time': elapsed,
                'success': path is not None
            }
            
            # Print results
            if path:
                print(f"    ✓ Success!")
                print(f"    Path length: {len(path.x)} points")
                print(f"    Collision rate: {collision_rate:.2%}")
                print(f"    Computation time: {elapsed:.3f}s")
                if uncertainty:
                    print(f"    Uncertainty range: [{min(uncertainty):.3f}, {max(uncertainty):.3f}]")
            else:
                print(f"    ✗ Failed to find path")
        
        results[scenario_name] = scenario_results
    
    return results

def create_summary_statistics(results):
    """Generate summary statistics table"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    methods = ['naive', 'ensemble', 'learnable_cp']
    
    # Aggregate statistics
    for method in methods:
        successes = []
        collision_rates = []
        computation_times = []
        
        for scenario_results in results.values():
            if method in scenario_results:
                result = scenario_results[method]
                if result['success']:
                    successes.append(1)
                    collision_rates.append(result['collision_rate'])
                    computation_times.append(result['computation_time'])
                else:
                    successes.append(0)
        
        if successes:
            print(f"\n{method.upper()}:")
            print(f"  Success rate: {np.mean(successes)*100:.1f}%")
            if collision_rates:
                print(f"  Avg collision rate: {np.mean(collision_rates)*100:.2f}%")
                print(f"  Avg computation time: {np.mean(computation_times):.3f}s")
    
    # Calculate improvement
    naive_collisions = []
    cp_collisions = []
    
    for scenario_results in results.values():
        if 'naive' in scenario_results and scenario_results['naive']['success']:
            naive_collisions.append(scenario_results['naive']['collision_rate'])
        if 'learnable_cp' in scenario_results and scenario_results['learnable_cp']['success']:
            cp_collisions.append(scenario_results['learnable_cp']['collision_rate'])
    
    if naive_collisions and cp_collisions:
        naive_avg = np.mean(naive_collisions)
        cp_avg = np.mean(cp_collisions)
        if naive_avg > 0:
            improvement = (naive_avg - cp_avg) / naive_avg * 100
            print(f"\n🎯 Collision Reduction (Naive → Learnable CP): {improvement:.1f}%")

def main():
    """Main execution"""
    
    print("="*80)
    print("ICRA 2025: LEARNABLE CONFORMAL PREDICTION FOR PATH PLANNING")
    print("Demonstration with Existing Hybrid A* Planner")
    print("="*80)
    
    # Run experiments
    results = run_scenario_comparison()
    
    # Create visualizations
    print("\nGenerating publication-quality visualizations...")
    visualize_comparison(results)
    
    # Print statistics
    create_summary_statistics(results)
    
    print("\n✅ Demonstration completed successfully!")
    print("📊 Results saved to: icra_implementation/phase2/results/")

if __name__ == "__main__":
    main()