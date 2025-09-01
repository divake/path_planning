#!/usr/bin/env python3
"""Debug exact hanging point"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))
sys.path.append(os.path.dirname(__file__))

from dataset_generation.generate_dataset_improved import ImprovedDatasetGenerator
import time
import numpy as np

print("Creating generator...")
gen = ImprovedDatasetGenerator(num_samples=2, robot_radius=0.5)

print("Starting generation...")
print(f"  Samples to generate: {gen.num_samples}")

env_types = ['walls', 'corridor', 'maze', 'random']

for trial_id in range(2):
    print(f"\n=== Trial {trial_id} ===")
    
    # Vary environment type
    env_type = env_types[trial_id % len(env_types)]
    print(f"1. Creating {env_type} environment...")
    env = gen.create_diverse_environment(env_type)
    base_obstacles = env.obstacles
    print(f"   Created with {len(base_obstacles)} obstacles")
    
    # Random start and goal
    print("2. Finding valid start/goal...")
    attempts = 0
    found = False
    while attempts < 100:
        start = (np.random.randint(5, 15), np.random.randint(5, 26))
        goal = (np.random.randint(35, 46), np.random.randint(5, 26))
        if start not in base_obstacles and goal not in base_obstacles:
            found = True
            print(f"   Found after {attempts} attempts: {start} -> {goal}")
            break
        attempts += 1
    
    if not found:
        print("   FAILED to find valid start/goal!")
        continue
    
    # Random noise level
    import numpy as np
    noise_level = np.random.choice(gen.noise_levels)
    print(f"3. Adding noise (level={noise_level})...")
    
    # Add realistic noise
    print("   Calling add_realistic_noise...")
    start_time = time.time()
    perceived_obstacles = gen.add_realistic_noise(
        base_obstacles, noise_level, robot_position=start
    )
    print(f"   Done in {time.time()-start_time:.2f}s, {len(perceived_obstacles)} perceived obstacles")
    
    # Plan path with A*
    print("4. Planning path...")
    import python_motion_planning as pmp
    perceived_env = pmp.Grid(51, 31)
    perceived_env.update(perceived_obstacles)
    factory = pmp.SearchFactory()
    planner = factory('a_star', start=start, goal=goal, env=perceived_env)
    
    try:
        start_time = time.time()
        print("   Calling planner.plan()...")
        cost, path, expand = planner.plan()
        print(f"   Path found in {time.time()-start_time:.2f}s: {len(path)} points")
    except Exception as e:
        print(f"   Planning failed: {e}")
        continue
    
    if not path or len(path) < 3:
        print("   Path too short, skipping")
        continue
    
    # Process each point
    print(f"5. Processing {len(path)} path points...")
    for i, point in enumerate(path[:3]):  # Just test first 3 points
        print(f"   Point {i}: {point}")
        
        # Extract features
        print(f"     Extracting features...")
        start_time = time.time()
        features = gen.extract_features_fast(
            point, path, i, perceived_obstacles, base_obstacles
        )
        print(f"     Features extracted in {time.time()-start_time:.3f}s: shape={features.shape}")
        
        # Compute risk
        print(f"     Computing risk...")
        start_time = time.time()
        risk_score = gen.compute_collision_risk(point, base_obstacles)
        print(f"     Risk computed in {time.time()-start_time:.3f}s: {risk_score:.3f}")

print("\nDebug complete!")