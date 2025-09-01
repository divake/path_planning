"""
Show WHY naive method collides - perception vs reality mismatch
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/ssd1/divake/path_planning/python_motion_planning/src')
from python_motion_planning.utils import Grid, SearchFactory


def create_wall_environment():
    """Create the wall environment."""
    grid = Grid(51, 31)
    obstacles = grid.obstacles
    
    for i in range(10, 21):
        obstacles.add((i, 15))
    for i in range(15):
        obstacles.add((20, i))
    for i in range(15, 30):
        obstacles.add((30, i))
    for i in range(16):
        obstacles.add((40, i))
    
    grid.update(obstacles)
    return grid


def demonstrate_perception_reality_mismatch():
    """
    Show why naive method fails - it plans on incorrect perception.
    """
    env = create_wall_environment()
    true_obstacles = env.obstacles
    
    # Create noisy perception (this is what the robot "sees")
    np.random.seed(42)  # For reproducibility
    perceived_obstacles = set()
    
    missing_obstacles = []  # Track which obstacles we missed
    shifted_obstacles = []  # Track perception errors
    
    for obs in true_obstacles:
        # 10% chance to completely miss an obstacle (sensor failure)
        if np.random.random() < 0.1:
            missing_obstacles.append(obs)
            continue
        
        # Add noise to position
        noise_x = np.random.normal(0, 0.3)
        noise_y = np.random.normal(0, 0.3)
        
        new_x = int(np.clip(obs[0] + noise_x, 0, 50))
        new_y = int(np.clip(obs[1] + noise_y, 0, 30))
        
        perceived_obstacles.add((new_x, new_y))
        if abs(noise_x) > 0.5 or abs(noise_y) > 0.5:
            shifted_obstacles.append((obs, (new_x, new_y)))
    
    # Plan based on PERCEIVED obstacles
    perceived_env = Grid(51, 31)
    perceived_env.update(perceived_obstacles)
    
    factory = SearchFactory()
    planner = factory('a_star', start=(5, 15), goal=(45, 15), env=perceived_env)
    
    try:
        cost, planned_path, _ = planner.plan()
    except:
        print("Planning failed")
        return None, None, None, None
    
    # Check where the path ACTUALLY collides with TRUE obstacles
    collision_points = []
    first_collision = None
    
    for i, point in enumerate(planned_path):
        for obs in true_obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            if dist < 0.5:  # Robot radius
                collision_points.append((i, point, obs))
                if first_collision is None:
                    first_collision = (i, point, obs)
    
    return planned_path, perceived_obstacles, missing_obstacles, collision_points, first_collision


def visualize_perception_vs_reality():
    """
    Create detailed visualization showing why collisions happen.
    """
    result = demonstrate_perception_reality_mismatch()
    if result[0] is None:
        print("Could not generate example")
        return
    
    planned_path, perceived_obs, missing_obs, collisions, first_collision = result
    
    env = create_wall_environment()
    true_obstacles = env.obstacles
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: What the robot PERCEIVES (noisy sensors)
    ax = axes[0]
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('What Robot PERCEIVES\n(Noisy/Missing Obstacles)', fontweight='bold', color='blue')
    
    # Show perceived obstacles
    for obs in perceived_obs:
        ax.plot(obs[0], obs[1], 'bs', markersize=4, alpha=0.6)
    
    # Show planned path
    if planned_path:
        path_x = [p[0] for p in planned_path]
        path_y = [p[1] for p in planned_path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Planned Path')
    
    # Highlight gaps in perception
    for obs in missing_obs:
        circle = plt.Circle((obs[0], obs[1]), 1, color='red', 
                           fill=False, linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.text(obs[0], obs[1]-2, 'MISSED!', fontsize=8, 
               color='red', ha='center', fontweight='bold')
    
    ax.plot(5, 15, 'go', markersize=10, label='Start')
    ax.plot(45, 15, 'ro', markersize=10, label='Goal')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 2: The REALITY (true obstacles)
    ax = axes[1]
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('The REALITY\n(True Obstacle Positions)', fontweight='bold', color='black')
    
    # Show true obstacles
    for obs in true_obstacles:
        ax.plot(obs[0], obs[1], 'ks', markersize=4)
    
    # Show the planned path
    if planned_path:
        path_x = [p[0] for p in planned_path]
        path_y = [p[1] for p in planned_path]
        ax.plot(path_x, path_y, 'g--', linewidth=2, alpha=0.5, 
               label='Planned Path (from perception)')
    
    # Mark ALL collision points
    for _, point, obs in collisions:
        ax.plot(point[0], point[1], 'rx', markersize=8, markeredgewidth=2)
    
    ax.plot(5, 15, 'go', markersize=10, label='Start')
    ax.plot(45, 15, 'ro', markersize=10, label='Goal')
    ax.legend(loc='upper right', fontsize=8)
    
    # Panel 3: Why it fails - Overlay
    ax = axes[2]
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('WHY NAIVE FAILS\n(Perception ≠ Reality)', fontweight='bold', color='red')
    
    # Show both perceived and true
    for obs in perceived_obs:
        ax.plot(obs[0], obs[1], 'b^', markersize=4, alpha=0.4, label='Perceived' if obs == list(perceived_obs)[0] else '')
    for obs in true_obstacles:
        ax.plot(obs[0], obs[1], 'ks', markersize=4, alpha=0.8, label='True' if obs == list(true_obstacles)[0] else '')
    
    # Show planned path
    if planned_path and first_collision:
        collision_idx, collision_point, collision_obs = first_collision
        
        # Path up to collision
        safe_path = planned_path[:collision_idx]
        if safe_path:
            safe_x = [p[0] for p in safe_path]
            safe_y = [p[1] for p in safe_path]
            ax.plot(safe_x, safe_y, 'g-', linewidth=2, label='Executed path')
        
        # Collision point
        ax.plot(collision_point[0], collision_point[1], 'rx', 
               markersize=15, markeredgewidth=3, label='COLLISION!')
        
        # Show robot at collision
        circle = plt.Circle((collision_point[0], collision_point[1]), 0.5, 
                          color='red', alpha=0.3)
        ax.add_patch(circle)
        
        # Draw arrow showing the mismatch
        if collision_obs in missing_obs:
            ax.annotate('Obstacle not detected!', 
                       xy=(collision_obs[0], collision_obs[1]),
                       xytext=(collision_obs[0]-5, collision_obs[1]+3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')
        else:
            ax.annotate('Perception error!', 
                       xy=(collision_obs[0], collision_obs[1]),
                       xytext=(collision_obs[0]-5, collision_obs[1]+3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')
    
    ax.plot(5, 15, 'go', markersize=10, label='Start')
    ax.plot(45, 15, 'ro', markersize=10, label='Goal')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Why Naive Method Collides: Perception vs Reality Mismatch', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/why_naive_collides.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to results/why_naive_collides.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("COLLISION ANALYSIS")
    print("="*60)
    print(f"Total collision points: {len(collisions)}")
    if first_collision:
        idx, point, obs = first_collision
        print(f"First collision at: step {idx}, position ({point[0]}, {point[1]})")
        print(f"Collided with obstacle at: ({obs[0]}, {obs[1]})")
        
        if obs in missing_obs:
            print("Reason: This obstacle was NOT DETECTED by sensors!")
        else:
            print("Reason: Perception error - obstacle position was wrong")


def main():
    print("="*60)
    print("WHY NAIVE METHOD COLLIDES")
    print("="*60)
    print("\nThe naive method plans based on PERCEIVED obstacles")
    print("But executes in REALITY with TRUE obstacles")
    print("Sensor noise/failures cause perception ≠ reality")
    print("Result: COLLISION!\n")
    
    visualize_perception_vs_reality()
    
    print("\n" + "="*60)
    print("KEY INSIGHT")
    print("="*60)
    print("Naive method fails because:")
    print("1. Sensors miss some obstacles (detection failure)")
    print("2. Sensors misplace obstacle positions (noise)")
    print("3. Path planned on wrong map → hits real obstacles")
    print("\nStandard CP fixes this by adding safety margins!")


if __name__ == "__main__":
    main()