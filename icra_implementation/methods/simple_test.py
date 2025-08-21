#!/usr/bin/env python3
"""
Simple test of Hybrid A* planner with known working scenario
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

def create_boundary_obstacles(width=60, height=60):
    """Create boundary walls as obstacles"""
    ox, oy = [], []
    
    # Bottom and top walls
    for i in range(width):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(float(height-1))
    
    # Left and right walls
    for i in range(height):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(float(width-1))
        oy.append(float(i))
    
    # Add some internal obstacles
    # Vertical wall with gap
    for i in range(15, 40):
        if i < 25 or i > 30:  # Gap from 25 to 30
            ox.append(30.0)
            oy.append(float(i))
    
    return ox, oy

def test_simple_path():
    """Test a simple path planning scenario"""
    
    # Create obstacles
    ox, oy = create_boundary_obstacles()
    
    # Start and goal
    sx, sy, syaw = 10.0, 10.0, np.deg2rad(0)
    gx, gy, gyaw = 50.0, 45.0, np.deg2rad(90)
    
    print(f"Start: ({sx:.1f}, {sy:.1f}, {np.rad2deg(syaw):.1f}°)")
    print(f"Goal: ({gx:.1f}, {gy:.1f}, {np.rad2deg(gyaw):.1f}°)")
    print(f"Obstacles: {len(ox)} boundary points")
    
    config = C()
    
    try:
        path = hybrid_astar_planning(
            sx, sy, syaw,
            gx, gy, gyaw,
            ox, oy,
            config.XY_RESO,
            config.YAW_RESO
        )
        
        if path:
            print(f"✓ Path found with {len(path.x_list)} waypoints")
            print(f"  Path cost: {path.cost:.2f}")
            print(f"  First few points:")
            for i in range(min(5, len(path.x_list))):
                p = path.x_list[i]
                print(f"    {i}: ({p.x:.2f}, {p.y:.2f}, {np.rad2deg(p.yaw):.1f}°)")
            return True
        else:
            print("✗ No path found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_path()
    
    if success:
        print("\n=== TEST PASSED ===")
    else:
        print("\n=== TEST FAILED ===")
        
        # Try a simpler scenario
        print("\nTrying simpler scenario...")
        ox = [float(i) for i in range(60)]
        oy = [0.0] * 60
        ox.extend([float(i) for i in range(60)])
        oy.extend([59.0] * 60)
        ox.extend([0.0] * 60)
        oy.extend([float(i) for i in range(60)])
        ox.extend([59.0] * 60)
        oy.extend([float(i) for i in range(60)])
        
        sx, sy, syaw = 10.0, 10.0, 0.0
        gx, gy, gyaw = 50.0, 50.0, 0.0
        
        config = C()
        path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, config.XY_RESO, config.YAW_RESO)
        
        if path:
            print(f"✓ Simple path found with {len(path.x_list)} waypoints")
        else:
            print("✗ Even simple scenario failed")