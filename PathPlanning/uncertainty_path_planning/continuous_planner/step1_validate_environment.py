#!/usr/bin/env python3
"""
Step 1: Validate Basic Environment Setup
Let's ensure the environment, obstacles, and collision checking work correctly
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from continuous_environment import ContinuousEnvironment

def test_environment_creation():
    """Test 1: Can we create an environment with obstacles?"""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)
    
    # Create environment with different types
    env_types = ["open", "passages", "cluttered", "maze", "narrow"]
    
    for env_type in env_types:
        env = ContinuousEnvironment(env_type=env_type)
        print(f"\n✓ Environment created: {env_type}")
        print(f"  Width: {env.width}")
        print(f"  Height: {env.height}")
        print(f"  Bounds: {env.bounds}")
        print(f"  Number of obstacles: {len(env.obstacles)}")
    
    # Return one for further testing
    env = ContinuousEnvironment(env_type="open")
    return env

def test_collision_checking(env):
    """Test 2: Does collision checking work correctly?"""
    print("\n" + "="*60)
    print("TEST 2: Collision Checking")
    print("="*60)
    
    # Test points that should be free (in open environment)
    free_points = [
        [2, 2],     # Near corner
        [25, 15],   # Center area
        [5, 15],    # Left side
        [45, 25],   # Top right
    ]
    
    # Test points that should collide (based on open environment obstacles)
    # Open env has obstacles at: (12, 8, 8, 6), (30, 5, 6, 8), (8, 18, 10, 4), (35, 20, 8, 4)
    collision_points = [
        [16, 11],   # Inside central obstacle (12+4, 8+3)
        [33, 9],    # Inside right obstacle (30+3, 5+4)
        [13, 20],   # Inside top left obstacle (8+5, 18+2)
        [39, 22],   # Inside top right obstacle (35+4, 20+2)
    ]
    
    print("Testing free points:")
    for point in free_points:
        collision = env.point_in_obstacle(point[0], point[1])
        status = "✗ COLLISION" if collision else "✓ FREE"
        print(f"  Point {point}: {status}")
        if collision:
            print("    ERROR: This point should be free!")
    
    print("\nTesting collision points:")
    for point in collision_points:
        collision = env.point_in_obstacle(point[0], point[1])
        status = "✓ COLLISION" if collision else "✗ FREE"
        print(f"  Point {point}: {status}")
        if not collision:
            print("    ERROR: This point should collide!")

def test_path_collision(env):
    """Test 3: Can we check collision for a path?"""
    print("\n" + "="*60)
    print("TEST 3: Path Collision Checking")
    print("="*60)
    
    # Path that should be free (goes around obstacles in open env)
    free_path = np.array([
        [2, 2],
        [5, 5],
        [10, 5],
        [10, 15],
        [25, 15],
        [25, 25],
        [45, 25]
    ])
    
    # Path that goes through obstacle
    collision_path = np.array([
        [2, 2],
        [16, 11],  # Through central obstacle
        [45, 25]
    ])
    
    print("Testing free path:")
    free_collisions = 0
    for i, point in enumerate(free_path):
        if env.point_in_obstacle(point[0], point[1]):
            free_collisions += 1
            print(f"  ✗ Point {i} at {point} collides")
    
    if free_collisions == 0:
        print(f"  ✓ All {len(free_path)} points are collision-free")
    else:
        print(f"  ERROR: {free_collisions}/{len(free_path)} points collide!")
    
    print("\nTesting collision path:")
    collision_count = 0
    for i, point in enumerate(collision_path):
        if env.point_in_obstacle(point[0], point[1]):
            collision_count += 1
            print(f"  ✓ Point {i} at {point} collides (expected)")
    
    print(f"  Result: {collision_count}/{len(collision_path)} points collide")

def visualize_environment(env):
    """Test 4: Visualize the environment"""
    print("\n" + "="*60)
    print("TEST 4: Environment Visualization")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw obstacles (obstacles are in format: (x, y, width, height))
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = plt.Rectangle((x, y), w, h,
                            facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=2)
        ax.add_patch(rect)
    
    # Define typical start and goal for open environment
    start = [2, 2]
    goal = [48, 28]
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'b*', markersize=20, label='Goal')
    
    # Test grid of points for collision
    x = np.linspace(0, env.width, 50)
    y = np.linspace(0, env.height, 50)
    
    free_points = []
    collision_points = []
    
    for xi in x:
        for yi in y:
            if env.point_in_obstacle(xi, yi):
                collision_points.append([xi, yi])
            else:
                free_points.append([xi, yi])
    
    # Plot collision map
    if free_points:
        free_points = np.array(free_points)
        ax.scatter(free_points[:, 0], free_points[:, 1], c='lightgreen', s=5, alpha=0.3)
    
    if collision_points:
        collision_points = np.array(collision_points)
        ax.scatter(collision_points[:, 0], collision_points[:, 1], c='red', s=5, alpha=0.3)
    
    ax.set_xlim(-0.5, env.width + 0.5)
    ax.set_ylim(-0.5, env.height + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Environment Validation: Obstacles and Collision Map')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('step1_environment_validation.png', dpi=150)
    print(f"  ✓ Visualization saved to step1_environment_validation.png")
    plt.show()

def main():
    """Run all validation tests"""
    print("\n" + "#"*60)
    print("# STEP 1: VALIDATING ENVIRONMENT SETUP")
    print("#"*60)
    
    # Test 1: Create environment
    env = test_environment_creation()
    
    # Test 2: Collision checking
    test_collision_checking(env)
    
    # Test 3: Path collision
    test_path_collision(env)
    
    # Test 4: Visualization
    visualize_environment(env)
    
    print("\n" + "#"*60)
    print("# VALIDATION COMPLETE")
    print("#"*60)
    print("\nIf all tests passed, the environment is working correctly.")
    print("Next step: Validate the RRT* planner.")

if __name__ == "__main__":
    main()