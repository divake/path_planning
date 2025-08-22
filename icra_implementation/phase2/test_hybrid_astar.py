#!/usr/bin/env python3
"""
Test if we can import and run the basic Hybrid A* planner
"""

import numpy as np
import sys
import os

# Add path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Testing Hybrid A* import...")

try:
    from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Simple test scenario
print("\nCreating simple test scenario...")
ox = []
oy = []

# Boundaries  
for i in range(60):
    ox.append(float(i))
    oy.append(0.0)
    ox.append(float(i))
    oy.append(40.0)

for i in range(40):
    ox.append(0.0)
    oy.append(float(i))
    ox.append(60.0)
    oy.append(float(i))

# Simple obstacle
for i in range(20, 30):
    ox.append(float(i))
    oy.append(20.0)

sx, sy, syaw = 10.0, 10.0, np.deg2rad(0.0)
gx, gy, gyaw = 50.0, 30.0, np.deg2rad(0.0)

print(f"Start: ({sx}, {sy}, {np.rad2deg(syaw)}°)")
print(f"Goal: ({gx}, {gy}, {np.rad2deg(gyaw)}°)")
print(f"Obstacles: {len(ox)} points")

# Test planning
print("\nTesting path planning...")
config = C()
print(f"XY Resolution: {config.XY_RESO}")
print(f"Yaw Resolution: {np.rad2deg(config.YAW_RESO)}°")

try:
    print("Calling hybrid_astar_planning...")
    path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                                config.XY_RESO, config.YAW_RESO)
    
    if path:
        print(f"✓ Path found! Length: {len(path.x)} points")
        print(f"  First point: ({path.x[0]:.2f}, {path.y[0]:.2f})")
        print(f"  Last point: ({path.x[-1]:.2f}, {path.y[-1]:.2f})")
    else:
        print("✗ No path found")
        
except Exception as e:
    print(f"✗ Planning failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test completed")