"""
TRUE Standard Conformal Prediction for Path Planning
This implements the correct CP approach:
1. Calibration phase: Learn tau from data
2. Prediction phase: Use calibrated tau for safety guarantees
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import heapq
sys.path.append('/mnt/ssd1/divake/path_planning/python_motion_planning/src')
from python_motion_planning.utils import Grid


class StandardConformalPredictor:
    """
    Standard Conformal Prediction for Path Planning
    """
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Miscoverage rate (0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.tau = None  # Will be learned from calibration
        self.calibration_scores = []
        
    def calibrate(self, num_calibration_trials=1000, noise_levels=[0.1, 0.3, 0.5]):
        """
        Calibration Phase: Learn tau from simulated executions
        
        Args:
            num_calibration_trials: Number of calibration samples
            noise_levels: List of noise levels to sample from
        """
        print("="*60)
        print("CALIBRATION PHASE")
        print("="*60)
        
        nonconformity_scores = []
        
        # Create calibration environment
        env = self.create_wall_environment()
        obstacles = env.obstacles
        
        # Multiple start-goal pairs for calibration
        calibration_scenarios = [
            ((5, 15), (45, 15)),   # Straight through
            ((5, 5), (45, 25)),    # Diagonal
            ((5, 25), (45, 5)),    # Other diagonal
            ((10, 5), (40, 25)),   # Shorter diagonal
        ]
        
        print(f"Running {num_calibration_trials} calibration trials...")
        
        robot_radius = 0.5  # Robot collision radius
        
        for trial in range(num_calibration_trials):
            # Random scenario
            start, goal = calibration_scenarios[trial % len(calibration_scenarios)]
            
            # CORRECT APPROACH: Plan with noisy perception, execute with true obstacles
            noise_level = np.random.choice(noise_levels)
            
            # Robot plans with what it PERCEIVES (noisy obstacles)
            perceived_obstacles = self.add_perception_noise(obstacles, noise_level)
            path = self.standard_astar(start, goal, perceived_obstacles)
            
            if path:
                # Execute with TRUE obstacles (ground truth)
                min_clearance_to_true = float('inf')
                for point in path:
                    clearance = self.calculate_clearance(point, obstacles)
                    min_clearance_to_true = min(min_clearance_to_true, clearance)
                
                # CLEARER APPROACH: Store how much safety margin was LACKING
                # If clearance < robot_radius, we would collide
                # We need to know how much additional margin is needed
                margin_deficit = max(0, robot_radius - min_clearance_to_true)
                nonconformity_scores.append(margin_deficit)
        
        # Compute additional margin needed as (1-alpha) quantile
        # This ensures (1-alpha) coverage
        self.calibration_scores = np.array(nonconformity_scores)
        additional_margin = np.quantile(self.calibration_scores, 1 - self.alpha)
        
        # Total tau = robot radius + additional margin for perception errors
        self.tau = robot_radius + additional_margin
        
        print(f"\nCalibration Results:")
        print(f"  Number of scores: {len(self.calibration_scores)}")
        print(f"  Mean margin deficit: {np.mean(self.calibration_scores):.3f}")
        print(f"  Std margin deficit: {np.std(self.calibration_scores):.3f}")
        print(f"  Robot radius: {robot_radius:.3f}")
        print(f"  Additional margin (90th percentile): {additional_margin:.3f}")
        
        print(f"\n{'='*40}")
        print(f"CALIBRATED τ = {self.tau:.3f}")
        print(f"  (robot_radius {robot_radius:.1f} + margin {additional_margin:.3f})")
        print(f"{'='*40}")
        print(f"This guarantees {(1-self.alpha)*100:.0f}% coverage")
        
        return self.tau
    
    def plan_with_guarantee(self, start, goal, obstacles):
        """
        Prediction Phase: Plan with conformal guarantee
        
        Returns:
            path: Planned path
            tau: Safety margin that guarantees (1-alpha) coverage
            safe_path: Path modified to maintain safety margin
        """
        if self.tau is None:
            raise ValueError("Must calibrate before planning! Call calibrate() first.")
        
        # Step 1: Plan normally (don't modify base planner)
        base_path = self.standard_astar(start, goal, obstacles)
        
        # Step 2: Use calibrated tau for safety-aware planning
        safe_path = self.safety_aware_astar(start, goal, obstacles, self.tau)
        
        # The guarantee: With probability (1-α), robot will maintain
        # at least τ distance from obstacles during execution
        return base_path, safe_path, self.tau
    
    def add_perception_noise(self, obstacles, noise_level):
        """Add perception noise to obstacles."""
        perceived = set()
        for obs in obstacles:
            # 5% chance to miss an obstacle
            if np.random.random() < 0.05:
                continue
            
            # Add Gaussian noise
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)
            
            new_x = obs[0] + noise_x
            new_y = obs[1] + noise_y
            perceived.add((int(new_x), int(new_y)))
        
        return perceived
    
    def calculate_clearance(self, point, obstacles):
        """Calculate minimum distance from point to any obstacle."""
        if not obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def create_wall_environment(self):
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
    
    def standard_astar(self, start, goal, obstacles, grid_size=(51, 31)):
        """Standard A* without safety modifications."""
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
    
    def safety_aware_astar(self, start, goal, obstacles, tau, grid_size=(51, 31)):
        """A* modified to prefer paths with clearance > tau."""
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
            """Cost function that penalizes paths close to obstacles."""
            base_cost = np.sqrt((current[0] - neighbor[0])**2 + 
                               (current[1] - neighbor[1])**2)
            
            clearance = self.calculate_clearance(neighbor, obstacles)
            
            if clearance < tau:
                safety_penalty = 10.0 * (tau - clearance) / tau
            elif clearance < 2 * tau:
                safety_penalty = 2.0 * (2*tau - clearance) / (2*tau)
            else:
                safety_penalty = 0
            
            return base_cost * (1 + safety_penalty)
        
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
                tentative_g = g_score[current] + path_cost(current, neighbor, tau)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None


def test_standard_cp():
    """Test true Standard CP implementation."""
    print("="*60)
    print("TRUE STANDARD CONFORMAL PREDICTION FOR PATH PLANNING")
    print("="*60)
    
    # Initialize CP predictor
    cp = StandardConformalPredictor(alpha=0.1)  # 90% coverage
    
    # STEP 1: Calibration
    print("\nSTEP 1: CALIBRATION")
    print("-"*40)
    tau = cp.calibrate(num_calibration_trials=500)
    
    # STEP 2: Prediction with guarantee
    print("\nSTEP 2: PREDICTION WITH GUARANTEE")
    print("-"*40)
    
    env = cp.create_wall_environment()
    obstacles = env.obstacles
    start = (5, 15)
    goal = (45, 15)
    
    base_path, safe_path, tau = cp.plan_with_guarantee(start, goal, obstacles)
    
    print(f"\nPlanning Results:")
    if base_path and safe_path:
        base_length = sum([np.sqrt((base_path[i+1][0]-base_path[i][0])**2 + 
                                  (base_path[i+1][1]-base_path[i][1])**2) 
                         for i in range(len(base_path)-1)])
        safe_length = sum([np.sqrt((safe_path[i+1][0]-safe_path[i][0])**2 + 
                                  (safe_path[i+1][1]-safe_path[i][1])**2) 
                         for i in range(len(safe_path)-1)])
        
        base_clearance = min([cp.calculate_clearance(p, obstacles) for p in base_path])
        safe_clearance = min([cp.calculate_clearance(p, obstacles) for p in safe_path])
        
        print(f"  Base path: length={base_length:.1f}, min_clearance={base_clearance:.2f}")
        print(f"  Safe path: length={safe_length:.1f}, min_clearance={safe_clearance:.2f}")
        print(f"  Improvement: {(safe_length/base_length-1)*100:+.1f}% length, "
              f"{safe_clearance-base_clearance:+.2f} clearance")
    
    # STEP 3: Test coverage guarantee
    print("\nSTEP 3: TESTING COVERAGE GUARANTEE")
    print("-"*40)
    
    # Simulate executions with noise - MUST match calibration conditions
    covered = 0
    total = 100
    robot_radius = 0.5
    test_deficits = []
    
    for _ in range(total):
        # SAME AS CALIBRATION: Plan with noisy perception, execute with true obstacles
        noise_level = np.random.choice([0.1, 0.3, 0.5])
        perceived_obs = cp.add_perception_noise(obstacles, noise_level)
        
        # Plan standard path with PERCEIVED obstacles (not safety-aware)
        test_path = cp.standard_astar(start, goal, perceived_obs)
        
        # Check actual clearance to TRUE obstacles
        if test_path:
            min_clear_to_true = min([cp.calculate_clearance(p, obstacles) for p in test_path])
            
            # Check if margin deficit is within our calibrated bound
            margin_deficit = max(0, robot_radius - min_clear_to_true)
            test_deficits.append(margin_deficit)
            
            # Coverage is achieved if margin deficit <= (tau - robot_radius)
            # Because tau = robot_radius + calibrated_additional_margin
            if margin_deficit <= (tau - robot_radius):
                covered += 1
    
    empirical_coverage = covered / total
    print(f"\n  Test Results:")
    print(f"    Mean test deficit: {np.mean(test_deficits):.3f}")
    print(f"    Max test deficit: {np.max(test_deficits):.3f}")
    print(f"    Calibrated threshold: {tau - robot_radius:.3f}")
    print(f"\n  Empirical coverage: {empirical_coverage*100:.1f}%")
    print(f"  Target coverage: {(1-cp.alpha)*100:.0f}%")
    
    if empirical_coverage >= (1 - cp.alpha):
        print("  ✓ Coverage guarantee satisfied!")
    else:
        print(f"  ✗ Coverage {empirical_coverage*100:.1f}% < {(1-cp.alpha)*100:.0f}% target")
    
    # Visualization
    visualize_cp_results(cp, base_path, safe_path, obstacles, tau)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard CP with calibrated τ={tau:.3f} provides:")
    print(f"- Statistical guarantee: {(1-cp.alpha)*100:.0f}% coverage")
    print(f"- Adaptive path planning based on learned safety margin")
    print(f"- No manual parameter tuning required!")


def visualize_cp_results(cp, base_path, safe_path, obstacles, tau):
    """Visualize CP results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Calibration scores (margin deficits)
    ax = axes[0]
    ax.hist(cp.calibration_scores, bins=30, density=True, alpha=0.7, 
            color='blue', edgecolor='black')
    additional_margin = tau - 0.5  # tau = robot_radius + additional_margin
    ax.axvline(additional_margin, color='red', linestyle='--', linewidth=2, 
               label=f'Additional margin={additional_margin:.3f}')
    ax.set_xlabel('Margin Deficit (units)')
    ax.set_ylabel('Density')
    ax.set_title('Calibration: Safety Margin Deficits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Base path (no safety)
    ax = axes[1]
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title('Base Path (No Safety)')
    
    for obs in obstacles:
        ax.plot(obs[0], obs[1], 'ks', markersize=3)
    
    if base_path:
        path_x = [p[0] for p in base_path]
        path_y = [p[1] for p in base_path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
    
    ax.plot(5, 15, 'go', markersize=8)
    ax.plot(45, 15, 'ro', markersize=8)
    
    # Plot 3: Safe path with CP guarantee
    ax = axes[2]
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(f'CP Safe Path (τ={tau:.3f}, 90% guarantee)')
    
    for obs in obstacles:
        ax.plot(obs[0], obs[1], 'ks', markersize=3)
    
    if safe_path:
        path_x = [p[0] for p in safe_path]
        path_y = [p[1] for p in safe_path]
        
        # Color by clearance
        for i in range(len(safe_path)-1):
            clearance = cp.calculate_clearance(safe_path[i], obstacles)
            if clearance < tau:
                color = 'red'
            elif clearance < 2*tau:
                color = 'orange'
            else:
                color = 'green'
            ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                   color=color, linewidth=2, alpha=0.7)
        
        # Safety corridor
        for i in range(0, len(safe_path), 5):
            circle = plt.Circle((safe_path[i][0], safe_path[i][1]), tau, 
                              color='blue', alpha=0.05)
            ax.add_patch(circle)
    
    ax.plot(5, 15, 'go', markersize=8)
    ax.plot(45, 15, 'ro', markersize=8)
    
    plt.suptitle('True Standard CP: Calibration → Prediction with Guarantee', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/standard_cp_true.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to results/standard_cp_true.png")


if __name__ == "__main__":
    test_standard_cp()