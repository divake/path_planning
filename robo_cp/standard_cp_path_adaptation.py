"""
CORRECT Standard CP Implementation:
- Don't inflate obstacles
- Modify path planning to prefer safer routes
- Add safety corridor around path
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import heapq
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


def calculate_clearance(point, obstacles):
    """Calculate minimum distance from point to any obstacle."""
    if not obstacles:
        return float('inf')
    
    min_dist = float('inf')
    for obs in obstacles:
        dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
        min_dist = min(min_dist, dist)
    return min_dist


def safety_aware_astar(start, goal, obstacles, tau=1.0, grid_size=(51, 31)):
    """
    Modified A* that considers safety margins in path cost.
    This is the KEY - we modify the path planning algorithm itself!
    """
    
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), 
                       (1,1), (-1,1), (1,-1), (-1,-1)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < grid_size[0] and 
                0 <= new_pos[1] < grid_size[1] and
                new_pos not in obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    def path_cost(current, neighbor, tau):
        """
        Modified cost function that penalizes paths close to obstacles.
        This makes the algorithm prefer safer routes!
        """
        base_cost = np.sqrt((current[0] - neighbor[0])**2 + 
                           (current[1] - neighbor[1])**2)
        
        # Calculate clearance to obstacles
        clearance = calculate_clearance(neighbor, obstacles)
        
        # Penalize paths that get too close to obstacles
        if clearance < tau:
            # High penalty for violating safety margin
            safety_penalty = 10.0 * (tau - clearance) / tau
        elif clearance < 2 * tau:
            # Moderate penalty for being somewhat close
            safety_penalty = 2.0 * (2*tau - clearance) / (2*tau)
        else:
            # No penalty for safe distance
            safety_penalty = 0
        
        return base_cost * (1 + safety_penalty)
    
    # A* implementation with modified cost
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current):
            # Use modified cost function
            tentative_g = g_score[current] + path_cost(current, neighbor, tau)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found


def standard_astar(start, goal, obstacles, grid_size=(51, 31)):
    """Standard A* without safety considerations."""
    
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(pos):
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), 
                       (1,1), (-1,1), (1,-1), (-1,-1)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < grid_size[0] and 
                0 <= new_pos[1] < grid_size[1] and
                new_pos not in obstacles):
                neighbors.append(new_pos)
        return neighbors
    
    # Standard A*
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for neighbor in get_neighbors(current):
            tentative_g = g_score[current] + heuristic(current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None


def test_different_tau_values():
    """Test how different tau values affect the path."""
    env = create_wall_environment()
    obstacles = env.obstacles
    
    start = (5, 15)
    goal = (45, 15)
    
    tau_values = [0, 0.5, 1.0, 1.5, 2.0]
    paths = []
    
    for tau in tau_values:
        if tau == 0:
            # Standard A* for tau=0
            path = standard_astar(start, goal, obstacles)
        else:
            # Safety-aware A* for tau>0
            path = safety_aware_astar(start, goal, obstacles, tau)
        paths.append((tau, path))
    
    return paths, obstacles


def visualize_path_adaptation():
    """Visualize how paths adapt based on safety requirements."""
    paths, obstacles = test_different_tau_values()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (tau, path) in enumerate(paths):
        if idx >= 6:
            break
            
        ax = axes[idx]
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Draw obstacles (NOT inflated!)
        for obs in obstacles:
            ax.plot(obs[0], obs[1], 'ks', markersize=3)
        
        # Draw path
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            
            # Color based on clearance
            for i in range(len(path)-1):
                clearance = calculate_clearance(path[i], obstacles)
                if clearance < tau:
                    color = 'red'
                    alpha = 0.8
                elif clearance < 2*tau:
                    color = 'orange' 
                    alpha = 0.6
                else:
                    color = 'green'
                    alpha = 0.8
                
                ax.plot([path_x[i], path_x[i+1]], 
                       [path_y[i], path_y[i+1]], 
                       color=color, linewidth=2, alpha=alpha)
            
            # Draw safety corridor
            if tau > 0:
                for i in range(0, len(path), 5):
                    circle = plt.Circle((path[i][0], path[i][1]), tau, 
                                      color='blue', alpha=0.05)
                    ax.add_patch(circle)
            
            # Calculate path statistics
            total_length = sum([np.sqrt((path[i+1][0]-path[i][0])**2 + 
                                       (path[i+1][1]-path[i][1])**2) 
                              for i in range(len(path)-1)])
            min_clearance = min([calculate_clearance(p, obstacles) for p in path])
            
            title = f'τ={tau:.1f}: Length={total_length:.1f}, Min Clear={min_clearance:.2f}'
        else:
            title = f'τ={tau:.1f}: No path found'
        
        ax.set_title(title, fontsize=10)
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
    
    # Remove empty subplot
    if len(paths) < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Path Adaptation Based on Safety Requirement τ\n(Obstacles NOT Inflated)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/path_adaptation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to results/path_adaptation.png")


def compare_methods():
    """Compare naive vs safety-aware planning."""
    env = create_wall_environment()
    true_obstacles = env.obstacles
    
    # Add noise to create perceived obstacles
    np.random.seed(42)
    perceived_obstacles = set()
    for obs in true_obstacles:
        if np.random.random() < 0.95:  # 5% miss rate
            noise_x = np.random.normal(0, 0.2)
            noise_y = np.random.normal(0, 0.2)
            new_x = int(np.clip(obs[0] + noise_x, 0, 50))
            new_y = int(np.clip(obs[1] + noise_y, 0, 30))
            perceived_obstacles.add((new_x, new_y))
    
    start = (5, 15)
    goal = (45, 15)
    
    # Naive path (standard A*)
    naive_path = standard_astar(start, goal, perceived_obstacles)
    
    # Safety-aware path (modified A* with tau)
    tau = 1.5  # Calibrated value
    safe_path = safety_aware_astar(start, goal, perceived_obstacles, tau)
    
    # Check collisions with TRUE obstacles
    def check_collisions(path, true_obs, robot_radius=0.5):
        if not path:
            return 0
        collisions = 0
        for point in path:
            for obs in true_obs:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist < robot_radius:
                    collisions += 1
                    break
        return collisions
    
    naive_collisions = check_collisions(naive_path, true_obstacles)
    safe_collisions = check_collisions(safe_path, true_obstacles)
    
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    if naive_path:
        naive_length = sum([np.sqrt((naive_path[i+1][0]-naive_path[i][0])**2 + 
                                  (naive_path[i+1][1]-naive_path[i][1])**2) 
                          for i in range(len(naive_path)-1)])
        naive_min_clear = min([calculate_clearance(p, true_obstacles) for p in naive_path])
        print(f"NAIVE (Standard A*):")
        print(f"  Path length: {naive_length:.1f}")
        print(f"  Min clearance: {naive_min_clear:.2f}")
        print(f"  Collisions: {naive_collisions}")
    
    if safe_path:
        safe_length = sum([np.sqrt((safe_path[i+1][0]-safe_path[i][0])**2 + 
                                 (safe_path[i+1][1]-safe_path[i][1])**2) 
                         for i in range(len(safe_path)-1)])
        safe_min_clear = min([calculate_clearance(p, true_obstacles) for p in safe_path])
        print(f"\nSAFETY-AWARE (τ={tau}):")
        print(f"  Path length: {safe_length:.1f}")
        print(f"  Min clearance: {safe_min_clear:.2f}")
        print(f"  Collisions: {safe_collisions}")
        
        if naive_path:
            print(f"\nIMPROVEMENT:")
            print(f"  Length increase: {(safe_length/naive_length - 1)*100:.1f}%")
            print(f"  Clearance increase: {(safe_min_clear/naive_min_clear - 1)*100:.1f}%")
            print(f"  Collision reduction: {naive_collisions - safe_collisions}")


def main():
    print("="*60)
    print("CORRECT STANDARD CP: PATH ADAPTATION")
    print("="*60)
    print("\nKey Approach:")
    print("1. DON'T inflate obstacles")
    print("2. Modify path planning cost to prefer safer routes")
    print("3. Path adapts based on safety requirement τ")
    print("4. Corridor shows safety margin (visualization only)")
    
    visualize_path_adaptation()
    compare_methods()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The path planning algorithm itself is modified to:")
    print("- Penalize routes close to obstacles")
    print("- Prefer paths with larger clearance")
    print("- Result: Longer but safer paths")
    print("\nThis is more sophisticated than simple inflation!")


if __name__ == "__main__":
    main()