#!/usr/bin/env python3
"""
Showcase demonstration - Single scenario with all three methods
Fast execution for immediate results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import time

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing modules...")
from phase2.uncertainty_wrapper import UncertaintyAwareHybridAstar
from phase2.complex_environments import create_parking_lot, get_scenario_start_goal
from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

def main():
    print("\n" + "="*70)
    print("ICRA 2025: LEARNABLE CONFORMAL PREDICTION")
    print("Single Scenario Showcase - Parking Lot Navigation")
    print("="*70)
    
    # Create scenario
    print("\nSetting up parking lot scenario...")
    ox, oy = create_parking_lot()
    sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal('parking_lot')
    
    print(f"  Start: ({sx:.1f}, {sy:.1f}) → Goal: ({gx:.1f}, {gy:.1f})")
    print(f"  Obstacles: {len(ox)} points")
    
    # Results storage
    results = {}
    
    # Method 1: Naive (baseline)
    print("\n1. NAIVE METHOD (Baseline Hybrid A*)...")
    start_time = time.time()
    try:
        config = C()
        naive_path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                         config.XY_RESO, config.YAW_RESO)
        naive_time = time.time() - start_time
        
        if naive_path:
            print(f"   ✓ Success! Path with {len(naive_path.x)} points")
            print(f"   Time: {naive_time:.3f}s")
            results['naive'] = {'path': naive_path, 'uncertainty': None, 'time': naive_time}
        else:
            print("   ✗ Failed")
            results['naive'] = None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['naive'] = None
    
    # Method 2: Ensemble
    print("\n2. ENSEMBLE METHOD...")
    start_time = time.time()
    try:
        planner = UncertaintyAwareHybridAstar(method='ensemble')
        ensemble_path, ensemble_uncertainty = planner.plan_with_uncertainty(
            sx, sy, syaw, gx, gy, gyaw, ox, oy)
        ensemble_time = time.time() - start_time
        
        if ensemble_path:
            print(f"   ✓ Success! Path with {len(ensemble_path.x)} points")
            print(f"   Time: {ensemble_time:.3f}s")
            print(f"   Uncertainty range: [{min(ensemble_uncertainty):.3f}, {max(ensemble_uncertainty):.3f}]")
            results['ensemble'] = {
                'path': ensemble_path, 
                'uncertainty': ensemble_uncertainty,
                'time': ensemble_time
            }
        else:
            print("   ✗ Failed")
            results['ensemble'] = None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['ensemble'] = None
    
    # Method 3: Learnable CP
    print("\n3. LEARNABLE CP METHOD...")
    start_time = time.time()
    try:
        planner = UncertaintyAwareHybridAstar(method='learnable_cp')
        cp_path, cp_uncertainty = planner.plan_with_uncertainty(
            sx, sy, syaw, gx, gy, gyaw, ox, oy)
        cp_time = time.time() - start_time
        
        if cp_path:
            print(f"   ✓ Success! Path with {len(cp_path.x)} points")
            print(f"   Time: {cp_time:.3f}s")
            print(f"   Uncertainty range: [{min(cp_uncertainty):.3f}, {max(cp_uncertainty):.3f}]")
            results['learnable_cp'] = {
                'path': cp_path,
                'uncertainty': cp_uncertainty,
                'time': cp_time
            }
        else:
            print("   ✗ Failed")
            results['learnable_cp'] = None
    except Exception as e:
        print(f"   ✗ Error: {e}")
        results['learnable_cp'] = None
    
    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ['naive', 'ensemble', 'learnable_cp']
    titles = ['Naive (Baseline)', 'Ensemble', 'Learnable CP']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (method, title, color) in enumerate(zip(methods, titles, colors)):
        ax = axes[idx]
        
        # Plot obstacles
        ax.scatter(ox, oy, c='black', s=0.5, alpha=0.3)
        
        # Plot path if available
        if method in results and results[method]:
            path = results[method]['path']
            uncertainty = results[method].get('uncertainty', [])
            
            # Draw uncertainty bands for non-naive methods
            if uncertainty and method != 'naive':
                for i in range(0, len(path.x), max(1, len(path.x)//20)):
                    radius = uncertainty[i] * 5  # Scale for visibility
                    circle = patches.Circle((path.x[i], path.y[i]), radius,
                                          color=color, alpha=0.1)
                    ax.add_patch(circle)
            
            # Draw path
            ax.plot(path.x, path.y, color=color, linewidth=2.5, alpha=0.9)
            
            # Draw vehicle at start and end
            config = C()
            for i, face_color in [(0, 'green'), (-1, 'red')]:
                rect = patches.Rectangle(
                    (path.x[i] - config.RB, path.y[i] - config.W/2),
                    config.RF + config.RB, config.W,
                    angle=np.rad2deg(path.yaw[i]),
                    facecolor=face_color, alpha=0.3
                )
                transform = patches.transforms.Affine2D().rotate_around(
                    path.x[i], path.y[i], path.yaw[i]) + ax.transData
                rect.set_transform(transform)
                ax.add_patch(rect)
        
        # Mark start and goal
        ax.plot(sx, sy, 'o', color='green', markersize=10, 
               markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(gx, gy, 's', color='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=2)
        
        # Formatting
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]' if idx == 0 else '', fontsize=11)
        ax.set_title(f'{title}\n', fontsize=12, fontweight='bold')
        
        if method in results and results[method]:
            time_str = f"Time: {results[method]['time']:.2f}s"
            if method != 'naive' and results[method]['uncertainty']:
                unc = results[method]['uncertainty']
                time_str += f"\nAdaptive uncertainty"
            ax.text(0.02, 0.98, time_str, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.2)
        ax.axis('equal')
        ax.set_xlim([-5, 105])
        ax.set_ylim([-5, 65])
    
    plt.suptitle('ICRA 2025: Learnable Conformal Prediction for Safe Path Planning\n'
                'Parking Lot Navigation with Adaptive Safety Margins',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs('icra_implementation/phase2/results', exist_ok=True)
    plt.savefig('icra_implementation/phase2/results/showcase.png', dpi=200, bbox_inches='tight')
    plt.savefig('icra_implementation/phase2/results/showcase.pdf', bbox_inches='tight')
    print("✓ Saved to icra_implementation/phase2/results/showcase.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for method, title in zip(methods, titles):
        if method in results and results[method]:
            print(f"\n{title}:")
            print(f"  Path found: Yes")
            print(f"  Computation time: {results[method]['time']:.3f}s")
            if method != 'naive':
                unc = results[method]['uncertainty']
                if unc:
                    print(f"  Adaptive uncertainty: Yes")
                    print(f"  Max uncertainty: {max(unc):.3f}")
        else:
            print(f"\n{title}: Failed")
    
    print("\n✅ Showcase completed successfully!")

if __name__ == "__main__":
    main()