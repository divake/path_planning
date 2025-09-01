#!/usr/bin/env python3
"""Debug dataset generation"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp

print("Testing basic setup...")

# Test 1: Can we create environments?
print("1. Creating Grid...")
grid = pmp.Grid(51, 31)
print("   Grid created successfully")

# Test 2: Can we add obstacles?
print("2. Adding obstacles...")
obstacles = grid.obstacles
for i in range(10, 21):
    obstacles.add((i, 15))
grid.update(obstacles)
print(f"   Added {len(obstacles)} obstacles")

# Test 3: Can we create planner?
print("3. Creating planner...")
factory = pmp.SearchFactory()
print("   Factory created")

# Test 4: Can we plan?
print("4. Planning path...")
start = (5, 15)
goal = (45, 15)
planner = factory('a_star', start=start, goal=goal, env=grid)
print("   Planner created")

print("5. Running plan()...")
try:
    cost, path, expand = planner.plan()
    print(f"   Path found: {len(path)} points")
except Exception as e:
    print(f"   Planning failed: {e}")

print("\n6. Testing environment types...")

# Test different environment types
def test_env_type(env_type):
    print(f"   Testing {env_type}...")
    env = pmp.Grid(51, 31)
    obstacles = env.obstacles
    
    if env_type == 'random':
        # This might be problematic
        num_obstacles = np.random.randint(20, 50)
        for _ in range(num_obstacles):
            x = np.random.randint(5, 46)
            y = np.random.randint(5, 26)
            obstacles.add((x, y))
            # Add clusters
            if np.random.random() < 0.3:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 5 <= x+dx <= 45 and 5 <= y+dy <= 25:
                            obstacles.add((x+dx, y+dy))
    
    env.update(obstacles)
    
    # Try to find valid start/goal
    attempts = 0
    found = False
    while attempts < 100:
        start = (np.random.randint(5, 15), np.random.randint(5, 26))
        goal = (np.random.randint(35, 46), np.random.randint(5, 26))
        if start not in obstacles and goal not in obstacles:
            found = True
            break
        attempts += 1
    
    if found:
        print(f"      Found valid start/goal after {attempts} attempts")
        # Try planning
        planner = factory('a_star', start=start, goal=goal, env=env)
        try:
            cost, path, expand = planner.plan()
            print(f"      Path found: {len(path)} points")
        except:
            print(f"      Planning failed")
    else:
        print(f"      Could not find valid start/goal!")

for env_type in ['walls', 'corridor', 'maze', 'random']:
    test_env_type(env_type)

print("\nAll tests completed!")