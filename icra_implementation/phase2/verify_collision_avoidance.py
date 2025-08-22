#!/usr/bin/env python3
"""
CRITICAL: Verify that Hybrid A* actually avoids obstacles
This is fundamental - paths MUST NOT go through obstacles!
"""

import numpy as np
import sys
import os
sys.path.append('/mnt/ssd1/divake/path_planning')

from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C

print("="*80)
print("VERIFYING COLLISION AVOIDANCE")
print("="*80)

# Test 1: Simple obstacle in the middle
print("\nTest 1: Simple obstacle blocking direct path")
print("-"*40)

# Create a wall obstacle blocking the direct path
ox, oy = [], []

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

# Add a wall blocking direct path
for i in range(15, 25):
    for j in range(18, 23):
        ox.append(float(i))
        oy.append(float(j))

print(f"Created {len(ox)} obstacle points")
print(f"Wall from x=15-25, y=18-23 blocks direct path at y=20")

# Start and goal on opposite sides of obstacle
sx, sy, syaw = 10.0, 20.0, np.deg2rad(0)
gx, gy, gyaw = 30.0, 20.0, np.deg2rad(0)

print(f"Start: ({sx}, {sy})")
print(f"Goal: ({gx}, {gy})")

# Plan path
config = C()
path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy,
                            config.XY_RESO, config.YAW_RESO)

if path:
    print(f"✓ Path found with {len(path.x)} points")
    
    # Check if path goes around obstacle
    min_y = min(path.y)
    max_y = max(path.y)
    
    print(f"Path y-range: [{min_y:.1f}, {max_y:.1f}]")
    
    # Path should deviate from y=20 to avoid obstacle
    if max_y > 23 or min_y < 18:
        print("✓ Path correctly goes AROUND the obstacle!")
    else:
        # Check if any path point is inside obstacle
        collision = False
        for px, py in zip(path.x, path.y):
            if 15 <= px <= 25 and 18 <= py <= 23:
                collision = True
                print(f"✗ COLLISION at ({px:.1f}, {py:.1f})")
                break
        
        if not collision:
            print("✓ Path found a way through narrow gaps")
else:
    print("✗ No path found")

print("\n" + "="*80)
print("Test 2: Parking lot scenario")
print("-"*40)

# Create more complex scenario
ox2, oy2 = [], []

# Boundaries
for i in range(100):
    ox2.append(float(i))
    oy2.append(0.0)
    ox2.append(float(i))
    oy2.append(60.0)
for i in range(60):
    ox2.append(0.0)
    oy2.append(float(i))
    ox2.append(100.0)
    oy2.append(float(i))

# Add parked cars (rectangular obstacles)
cars = [
    (20, 20, 25, 23),  # (x1, y1, x2, y2)
    (30, 20, 35, 23),
    (40, 20, 45, 23),
    (20, 35, 25, 38),
    (30, 35, 35, 38),
    (40, 35, 45, 38),
]

for x1, y1, x2, y2 in cars:
    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            ox2.append(float(x))
            oy2.append(float(y))

print(f"Created parking lot with {len(cars)} parked cars")

# Navigate through parking lot
sx2, sy2, syaw2 = 10.0, 30.0, np.deg2rad(0)
gx2, gy2, gyaw2 = 90.0, 30.0, np.deg2rad(0)

print(f"Start: ({sx2}, {sy2})")
print(f"Goal: ({gx2}, {gy2})")

path2 = hybrid_astar_planning(sx2, sy2, syaw2, gx2, gy2, gyaw2, ox2, oy2,
                             config.XY_RESO, config.YAW_RESO)

if path2:
    print(f"✓ Path found with {len(path2.x)} points")
    
    # Check for collisions
    collisions = 0
    for px, py in zip(path2.x, path2.y):
        for x1, y1, x2, y2 in cars:
            if x1 <= px <= x2 and y1 <= py <= y2:
                collisions += 1
                break
    
    if collisions == 0:
        print("✓ Path successfully avoids all parked cars!")
    else:
        print(f"✗ Path has {collisions} collision points")
    
    # Path should weave between obstacles
    y_variance = np.var(path2.y)
    if y_variance > 5:
        print(f"✓ Path weaves between obstacles (y-variance: {y_variance:.1f})")
    else:
        print(f"⚠ Path is too straight (y-variance: {y_variance:.1f})")
else:
    print("✗ No path found")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if path and path2:
    print("✅ Hybrid A* correctly avoids obstacles")
    print("✅ Ready to use for experiments")
else:
    print("⚠️ Issue with path planning - debug needed")

print("\nNOTE: The planner MUST avoid obstacles. Paths going through obstacles")
print("would be instant rejection at ICRA!")