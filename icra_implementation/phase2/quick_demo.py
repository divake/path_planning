#!/usr/bin/env python3
"""
Quick demo to test the uncertainty wrapper with existing Hybrid A*
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from uncertainty_wrapper import UncertaintyAwareHybridAstar
from complex_environments import create_parking_lot, get_scenario_start_goal
from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

def main():
    print("Testing Uncertainty-Aware Hybrid A* Wrapper...")
    
    # Get parking lot scenario
    ox, oy = create_parking_lot()
    sx, sy, syaw, gx, gy, gyaw = get_scenario_start_goal('parking_lot')
    
    print(f"Start: ({sx:.1f}, {sy:.1f}, {np.rad2deg(syaw):.1f}°)")
    print(f"Goal: ({gx:.1f}, {gy:.1f}, {np.rad2deg(gyaw):.1f}°)")
    print(f"Obstacles: {len(ox)} points")
    
    # Test naive method (baseline)
    print("\n1. Testing Naive Method (baseline)...")
    try:
        config = C()
        naive_path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                         config.XY_RESO, config.YAW_RESO)
        if naive_path:
            print(f"   Success! Path length: {len(naive_path.x)} points")
        else:
            print("   Failed to find path")
    except Exception as e:
        print(f"   Error: {e}")
        naive_path = None
    
    # Test learnable CP method
    print("\n2. Testing Learnable CP Method...")
    planner = UncertaintyAwareHybridAstar(method='learnable_cp')
    try:
        cp_path, uncertainty = planner.plan_with_uncertainty(sx, sy, syaw, gx, gy, gyaw, ox, oy)
        if cp_path:
            print(f"   Success! Path length: {len(cp_path.x)} points")
            print(f"   Uncertainty range: [{min(uncertainty):.3f}, {max(uncertainty):.3f}]")
        else:
            print("   Failed to find path")
    except Exception as e:
        print(f"   Error: {e}")
        cp_path = None
        uncertainty = []
    
    # Visualize results
    if naive_path or cp_path:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Naive method
        ax = axes[0]
        ax.plot(ox, oy, 'k.', markersize=1, alpha=0.5)
        if naive_path:
            ax.plot(naive_path.x, naive_path.y, 'b-', linewidth=2, label='Naive Path')
        ax.plot(sx, sy, 'go', markersize=10, label='Start')
        ax.plot(gx, gy, 'ro', markersize=10, label='Goal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Naive Method (Baseline)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot 2: Learnable CP method
        ax = axes[1]
        ax.plot(ox, oy, 'k.', markersize=1, alpha=0.5)
        if cp_path:
            ax.plot(cp_path.x, cp_path.y, 'b-', linewidth=2, label='CP Path')
            # Add uncertainty visualization
            for i in range(0, len(cp_path.x), max(1, len(cp_path.x)//20)):
                circle = plt.Circle((cp_path.x[i], cp_path.y[i]),
                                   uncertainty[i] * 3,  # Scale for visibility
                                   color='red', alpha=0.15)
                ax.add_patch(circle)
        ax.plot(sx, sy, 'go', markersize=10, label='Start')
        ax.plot(gx, gy, 'ro', markersize=10, label='Goal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Learnable CP Method')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig('icra_implementation/phase2/quick_demo.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nVisualization saved to quick_demo.png")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()