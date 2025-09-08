#!/usr/bin/env python3
"""
Step 2: Validate RRT* Planner
Let's ensure the RRT* planner works correctly and consistently
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append('.')

from continuous_environment import ContinuousEnvironment
from rrt_star_planner import RRTStar

def test_rrt_star_basic():
    """Test 1: Can RRT* find a path in a simple environment?"""
    print("\n" + "="*60)
    print("TEST 1: Basic RRT* Path Finding")
    print("="*60)
    
    # Create simple open environment
    env = ContinuousEnvironment(env_type="open")
    
    # Initialize RRT* planner
    planner = RRTStar(
        start=(2, 2),
        goal=(48, 28),
        obstacles=env.obstacles,
        bounds=(0, env.width, 0, env.height),
        robot_radius=0.5,
        step_size=2.0,
        max_iter=1000,
        seed=42
    )
    
    print(f"✓ RRT* planner initialized")
    print(f"  Start: ({planner.start.x}, {planner.start.y})")
    print(f"  Goal: ({planner.goal.x}, {planner.goal.y})")
    print(f"  Robot radius: {planner.robot_radius}")
    print(f"  Step size: {planner.step_size}")
    print(f"  Max iterations: {planner.max_iter}")
    
    # Plan path
    print("\nPlanning path...")
    start_time = time.time()
    path = planner.plan()
    planning_time = time.time() - start_time
    
    if path is not None:
        print(f"✓ Path found in {planning_time:.2f} seconds")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  Total distance: {compute_path_length(path):.2f}")
    else:
        print(f"✗ No path found after {planning_time:.2f} seconds")
    
    return env, planner, path

def test_rrt_star_multiple_runs():
    """Test 2: Check consistency across multiple runs"""
    print("\n" + "="*60)
    print("TEST 2: RRT* Consistency Check (5 runs)")
    print("="*60)
    
    env = ContinuousEnvironment(env_type="open")
    
    paths = []
    times = []
    lengths = []
    
    for i in range(5):
        planner = RRTStar(
            start=(2, 2),
            goal=(48, 28),
            obstacles=env.obstacles,
            bounds=(0, env.width, 0, env.height),
            robot_radius=0.5,
            step_size=2.0,
            max_iter=1000,
            seed=42 + i  # Different seed for each run
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path is not None:
            paths.append(path)
            times.append(planning_time)
            path_length = compute_path_length(path)
            lengths.append(path_length)
            print(f"  Run {i+1}: ✓ Found path with {len(path)} waypoints, "
                  f"length={path_length:.2f}, time={planning_time:.2f}s")
        else:
            print(f"  Run {i+1}: ✗ No path found")
    
    if len(paths) > 0:
        print(f"\nStatistics over {len(paths)} successful runs:")
        print(f"  Success rate: {len(paths)/5*100:.0f}%")
        print(f"  Avg path length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"  Avg planning time: {np.mean(times):.2f} ± {np.std(times):.2f} seconds")
        print(f"  Min/Max length: {np.min(lengths):.2f} / {np.max(lengths):.2f}")
    
    return paths

def test_rrt_star_collision_free():
    """Test 3: Verify paths are collision-free"""
    print("\n" + "="*60)
    print("TEST 3: Collision-Free Path Verification")
    print("="*60)
    
    env = ContinuousEnvironment(env_type="cluttered")
    
    planner = RRTStar(
        start=(2, 2),
        goal=(48, 28),
        obstacles=env.obstacles,
        bounds=(0, env.width, 0, env.height),
        robot_radius=0.5,
        step_size=2.0,
        max_iter=2000,
        seed=42
    )
    
    path = planner.plan()
    
    if path is None:
        print("✗ No path found")
        return None
    
    print(f"✓ Path found with {len(path)} waypoints")
    
    # Check each waypoint for collision
    collisions = 0
    for i, point in enumerate(path):
        if env.point_in_obstacle(point[0], point[1]):
            collisions += 1
            print(f"  ✗ Waypoint {i} at {point} is in collision!")
    
    # Check segments between waypoints
    segment_collisions = 0
    for i in range(len(path) - 1):
        segment_collision = check_segment_collision(env, path[i], path[i+1], 
                                                   planner.robot_radius)
        if segment_collision:
            segment_collisions += 1
            print(f"  ✗ Segment {i}->{i+1} has collision!")
    
    if collisions == 0 and segment_collisions == 0:
        print(f"  ✓ All {len(path)} waypoints are collision-free")
        print(f"  ✓ All {len(path)-1} segments are collision-free")
    else:
        print(f"  ERROR: Found {collisions} waypoint collisions and "
              f"{segment_collisions} segment collisions")
    
    return env, path

def test_rrt_star_different_environments():
    """Test 4: Test RRT* in different environment types"""
    print("\n" + "="*60)
    print("TEST 4: RRT* in Different Environments")
    print("="*60)
    
    env_types = ["open", "cluttered", "passages", "maze", "narrow"]
    results = {}
    
    for env_type in env_types:
        env = ContinuousEnvironment(env_type=env_type)
        
        planner = RRTStar(
            start=(2, 2),
            goal=(48, 28),
            obstacles=env.obstacles,
            bounds=(0, env.width, 0, env.height),
            robot_radius=0.5,
            step_size=2.0,
            max_iter=3000,
            seed=42
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path is not None:
            path_length = compute_path_length(path)
            results[env_type] = {
                'success': True,
                'waypoints': len(path),
                'length': path_length,
                'time': planning_time
            }
            print(f"  {env_type:10s}: ✓ {len(path):3d} waypoints, "
                  f"length={path_length:6.2f}, time={planning_time:.2f}s")
        else:
            results[env_type] = {
                'success': False,
                'time': planning_time
            }
            print(f"  {env_type:10s}: ✗ No path found after {planning_time:.2f}s")
    
    return results

def visualize_rrt_star_results(env, planner, path):
    """Visualize RRT* planning results"""
    print("\n" + "="*60)
    print("TEST 5: RRT* Visualization")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: RRT* tree
    ax1.set_title('RRT* Search Tree')
    
    # Draw obstacles
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = plt.Rectangle((x, y), w, h,
                            facecolor='red', alpha=0.5, edgecolor='darkred')
        ax1.add_patch(rect)
    
    # Draw RRT* tree if available
    if hasattr(planner, 'nodes') and planner.nodes:
        for node in planner.nodes:
            if node.parent:
                ax1.plot([node.parent.x, node.x], 
                        [node.parent.y, node.y], 
                        'b-', alpha=0.1, linewidth=0.5)
    
    # Draw start and goal
    ax1.plot(planner.start.x, planner.start.y, 'go', markersize=10, label='Start')
    ax1.plot(planner.goal.x, planner.goal.y, 'r*', markersize=15, label='Goal')
    
    # Draw path
    if path is not None:
        path = np.array(path)
        ax1.plot(path[:, 0], path[:, 1], 'g-', linewidth=3, label='Path')
        ax1.plot(path[:, 0], path[:, 1], 'go', markersize=3)
    
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right plot: Path with robot radius
    ax2.set_title('Path with Robot Radius')
    
    # Draw obstacles
    for obs in env.obstacles:
        x, y, w, h = obs
        rect = plt.Rectangle((x, y), w, h,
                            facecolor='red', alpha=0.5, edgecolor='darkred')
        ax2.add_patch(rect)
    
    # Draw path with robot radius
    if path is not None:
        path = np.array(path)
        ax2.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, label='Path')
        
        # Draw robot radius at each waypoint
        for point in path[::5]:  # Every 5th point to avoid clutter
            circle = plt.Circle((point[0], point[1]), planner.robot_radius,
                               facecolor='green', alpha=0.2, edgecolor='green')
            ax2.add_patch(circle)
    
    ax2.plot(planner.start.x, planner.start.y, 'go', markersize=10, label='Start')
    ax2.plot(planner.goal.x, planner.goal.y, 'r*', markersize=15, label='Goal')
    
    ax2.set_xlim(0, env.width)
    ax2.set_ylim(0, env.height)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('step2_rrt_star_validation.png', dpi=150)
    print(f"  ✓ Visualization saved to step2_rrt_star_validation.png")
    plt.show()

def compute_path_length(path):
    """Compute total path length"""
    if path is None or len(path) < 2:
        return 0.0
    
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += np.sqrt(dx*dx + dy*dy)
    return length

def check_segment_collision(env, p1, p2, robot_radius, num_checks=10):
    """Check if a line segment collides with obstacles"""
    for t in np.linspace(0, 1, num_checks):
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        
        # Check collision with robot radius
        for obs in env.obstacles:
            ox, oy, ow, oh = obs
            # Expand obstacle by robot radius
            if (ox - robot_radius <= x <= ox + ow + robot_radius and 
                oy - robot_radius <= y <= oy + oh + robot_radius):
                return True
    return False

def main():
    """Run all RRT* validation tests"""
    print("\n" + "#"*60)
    print("# STEP 2: VALIDATING RRT* PLANNER")
    print("#"*60)
    
    # Test 1: Basic functionality
    env, planner, path = test_rrt_star_basic()
    
    # Test 2: Consistency
    paths = test_rrt_star_multiple_runs()
    
    # Test 3: Collision-free paths
    env_cluttered, path_cluttered = test_rrt_star_collision_free()
    
    # Test 4: Different environments
    results = test_rrt_star_different_environments()
    
    # Test 5: Visualization
    if path is not None:
        visualize_rrt_star_results(env, planner, path)
    
    print("\n" + "#"*60)
    print("# VALIDATION COMPLETE")
    print("#"*60)
    print("\nIf all tests passed, the RRT* planner is working correctly.")
    print("Next step: Validate the noise model and perception uncertainty.")

if __name__ == "__main__":
    main()