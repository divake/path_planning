#!/usr/bin/env python3
"""
Learnable CP Main Evaluation Script
Integrates learnable scoring function with existing path planners.
Works with ANY planner - just change the base_planner in config.
"""

import numpy as np
import torch
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time

# Import existing modules
from mrpb_map_parser import MRPBMapParser
from standard_cp_noise_model import StandardCPNoiseModel
from standard_cp_nonconformity import StandardCPNonconformity
from rrt_star_grid_planner import RRTStarGrid

# Import learnable CP modules
from learn_cp_scoring_function import LearnableCPScoringFunction
from learn_cp_trainer import LearnableCPTrainer


class LearnableCPEvaluator:
    """
    Evaluator for Learnable CP.
    Works with any base planner by adjusting robot radius dynamically.
    """
    
    def __init__(self, config_path: str = "learn_cp_config.yaml"):
        """Initialize evaluator"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load environment configuration
        with open('config_env.yaml', 'r') as f:
            self.env_config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.learn_cp_model = None
        self.noise_model = StandardCPNoiseModel()
        self.nonconformity_calculator = StandardCPNonconformity()
        
        # Setup results directory
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Robot parameters
        self.robot_radius = 0.17  # Base robot radius (meters)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['debug']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_trained_model(self, checkpoint_path: str = None):
        """Load trained learnable CP model"""
        if checkpoint_path is None:
            checkpoint_path = self.results_dir / "checkpoints" / "best_model.pth"
        
        # Initialize model
        self.learn_cp_model = LearnableCPScoringFunction(self.config).to(self.device)
        
        # Load checkpoint
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.learn_cp_model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from {checkpoint_path}")
        else:
            self.logger.warning(f"No checkpoint found at {checkpoint_path}, using random initialization")
        
        self.learn_cp_model.eval()
    
    def plan_with_learnable_cp(self, env_name: str, test_id: int,
                               perceived_grid: np.ndarray,
                               true_grid: np.ndarray) -> Dict:
        """
        Plan a path using learnable CP for adaptive safety margins.
        
        Key innovation: Adjust robot radius based on local context
        """
        # Parse environment
        parser = MRPBMapParser(env_name)
        
        # Get test configuration
        test_config = self.env_config['environments'][env_name]['tests'][test_id]
        start = tuple(test_config['start'])
        goal = tuple(test_config['goal'])
        
        # Initialize grid-based planner (same as Standard CP)
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=perceived_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=self.robot_radius,  # Start with base radius
            max_iter=10000,  # Reduced for faster testing
            seed=42
        )
        
        # Plan initial path
        start_time = time.time()
        path = planner.plan()
        success = path is not None
        planning_time = time.time() - start_time
        
        if not success or path is None:
            return {
                'success': False,
                'collision': False,
                'path': None,
                'planning_time': planning_time,
                'tau_values': [],
                'adaptations': 0
            }
        
        # Convert path to numpy array if needed
        if isinstance(path, list):
            path = np.array(path)
        
        # Apply learnable CP adaptation
        adapted_path, tau_values = self.adapt_path_with_learnable_cp(
            path, perceived_grid, np.array(goal), parser.origin, parser.resolution
        )
        
        # Check collision on true environment
        collision = self.check_collision_grid(adapted_path, true_grid, parser.origin, parser.resolution)
        
        # Compute metrics
        path_length = self.compute_path_length(adapted_path)
        
        # Extract obstacles for clearance computation
        true_obstacles = np.array(parser.extract_obstacles())
        min_clearance = self.compute_min_clearance(adapted_path, true_obstacles)
        
        return {
            'success': True,
            'collision': collision,
            'path': adapted_path,
            'path_length': path_length,
            'min_clearance': min_clearance,
            'planning_time': planning_time,
            'tau_values': tau_values,
            'tau_mean': np.mean(tau_values) if tau_values else 0.0,
            'tau_std': np.std(tau_values) if tau_values else 0.0,
            'adaptations': len(tau_values)
        }
    
    def adapt_path_with_learnable_cp(self, path: np.ndarray, 
                                     grid: np.ndarray,
                                     goal: np.ndarray,
                                     origin: tuple,
                                     resolution: float) -> Tuple[np.ndarray, List[float]]:
        """
        Adapt path using learnable CP to add safety margins.
        
        Key idea: For each waypoint, predict adaptive tau and adjust local trajectory
        """
        if self.learn_cp_model is None:
            self.logger.warning("No learnable CP model loaded, returning original path")
            return path, []
        
        # Extract obstacles from grid for feature computation
        obstacles = self.extract_obstacles_from_grid(grid, origin, resolution)
        
        # Compute bounds from grid
        width_meters = grid.shape[1] * resolution
        height_meters = grid.shape[0] * resolution
        bounds = [origin[0], origin[0] + width_meters,
                  origin[1], origin[1] + height_meters]
        
        tau_values = []
        adapted_path = []
        
        for i, point in enumerate(path):
            # Predict tau for this location
            with torch.no_grad():
                tau = self.learn_cp_model.predict_tau(point, obstacles, goal, bounds)
            tau_values.append(tau)
            
            # Adapt point based on tau (push away from obstacles)
            if i > 0 and i < len(path) - 1:  # Don't modify start/goal
                # Find gradient away from nearest obstacle
                gradient = self.compute_safety_gradient(point, obstacles)
                
                # Apply adaptive adjustment
                adjusted_point = point + tau * gradient * 0.5  # Scale factor
                
                # Ensure within bounds
                adjusted_point[0] = np.clip(adjusted_point[0], bounds[0], bounds[1])
                adjusted_point[1] = np.clip(adjusted_point[1], bounds[2], bounds[3])
                
                adapted_path.append(adjusted_point)
            else:
                adapted_path.append(point)
        
        return np.array(adapted_path), tau_values
    
    def compute_safety_gradient(self, point: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Compute gradient pointing away from obstacles"""
        if len(obstacles) == 0:
            return np.zeros(2)
        
        # Find nearest obstacle
        min_dist = float('inf')
        nearest_obs = None
        for obs in obstacles:
            dist = self.point_to_rectangle_distance(point, obs)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs
        
        if nearest_obs is None:
            return np.zeros(2)
        
        # Compute gradient away from obstacle center
        obs_center = np.array([
            (nearest_obs[0] + nearest_obs[2]) / 2,
            (nearest_obs[1] + nearest_obs[3]) / 2
        ])
        
        gradient = point - obs_center
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient = gradient / norm
        
        return gradient
    
    def evaluate_all_methods(self):
        """
        Evaluate all methods: Naive, Standard CP, Learnable CP
        """
        results = {
            'naive': [],
            'standard_cp': [],
            'learnable_cp': []
        }
        
        # Test environments from config
        test_envs = self.config['test_environments']
        num_trials = self.config['evaluation']['num_trials']
        
        self.logger.info(f"Evaluating on {len(test_envs)} environments with {num_trials} trials each")
        
        for env_name in test_envs:
            self.logger.info(f"Testing environment: {env_name}")
            
            # Parse environment
            parser = MRPBMapParser(env_name)
            true_obstacles = np.array(parser.extract_obstacles())
            
            # Get test configurations
            env_config = self.env_config['environments'][env_name]
            
            for test_id, test in enumerate(env_config['tests'][:1]):  # Use first test for now
                self.logger.info(f"  Test {test_id}: start={test['start']}, goal={test['goal']}")
                
                # Run trials
                for trial in tqdm(range(num_trials), desc=f"{env_name} trials"):
                    # Get true grid
                    true_grid = parser.occupancy_grid.copy()
                    
                    # Add noise to create perceived grid (same as Standard CP)
                    perceived_grid = self.noise_model.add_realistic_noise(true_grid)
                    
                    # 1. Naive planning (no uncertainty)
                    naive_result = self.plan_naive(
                        env_name, test_id, perceived_grid, true_grid
                    )
                    naive_result['env_name'] = env_name
                    naive_result['trial'] = trial
                    results['naive'].append(naive_result)
                    
                    # 2. Standard CP (fixed tau = 0.17m)
                    standard_result = self.plan_standard_cp(
                        env_name, test_id, perceived_grid, true_grid
                    )
                    standard_result['env_name'] = env_name
                    standard_result['trial'] = trial
                    results['standard_cp'].append(standard_result)
                    
                    # 3. Learnable CP (adaptive tau)
                    learnable_result = self.plan_with_learnable_cp(
                        env_name, test_id, perceived_grid, true_grid
                    )
                    learnable_result['env_name'] = env_name
                    learnable_result['trial'] = trial
                    results['learnable_cp'].append(learnable_result)
        
        # Compute statistics
        statistics = self.compute_statistics(results)
        
        # Save results
        self.save_results(results, statistics)
        
        # Generate plots
        self.generate_comparison_plots(statistics)
        
        return statistics
    
    def plan_naive(self, env_name: str, test_id: int,
                  perceived_grid: np.ndarray,
                  true_grid: np.ndarray) -> Dict:
        """Naive planning without uncertainty consideration"""
        parser = MRPBMapParser(env_name)
        
        test_config = self.env_config['environments'][env_name]['tests'][test_id]
        start = tuple(test_config['start'])
        goal = tuple(test_config['goal'])
        
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=perceived_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=self.robot_radius,
            max_iter=50000,
            seed=42
        )
        
        start_time = time.time()
        path = planner.plan()
        success = path is not None
        planning_time = time.time() - start_time
        
        if not success or path is None:
            return {
                'success': False,
                'collision': False,
                'path': None,
                'planning_time': planning_time
            }
        
        collision = self.check_collision_grid(path, true_grid, parser.origin, parser.resolution)
        path_length = self.compute_path_length(path)
        
        # Extract obstacles for clearance computation
        true_obstacles = np.array(parser.extract_obstacles())
        min_clearance = self.compute_min_clearance(path, true_obstacles)
        
        return {
            'success': True,
            'collision': collision,
            'path': path,
            'path_length': path_length,
            'min_clearance': min_clearance,
            'planning_time': planning_time
        }
    
    def plan_standard_cp(self, env_name: str, test_id: int,
                        perceived_grid: np.ndarray,
                        true_grid: np.ndarray) -> Dict:
        """Standard CP planning with fixed tau = 0.17m"""
        parser = MRPBMapParser(env_name)
        
        test_config = self.env_config['environments'][env_name]['tests'][test_id]
        start = tuple(test_config['start'])
        goal = tuple(test_config['goal'])
        
        # Fixed tau from standard CP
        tau = 0.17
        
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=perceived_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=self.robot_radius + tau,  # Inflate radius
            max_iter=50000,
            seed=42
        )
        
        start_time = time.time()
        path = planner.plan()
        success = path is not None
        planning_time = time.time() - start_time
        
        if not success or path is None:
            return {
                'success': False,
                'collision': False,
                'path': None,
                'planning_time': planning_time,
                'tau': tau
            }
        
        collision = self.check_collision_grid(path, true_grid, parser.origin, parser.resolution)
        path_length = self.compute_path_length(path)
        
        # Extract obstacles for clearance computation
        true_obstacles = np.array(parser.extract_obstacles())
        min_clearance = self.compute_min_clearance(path, true_obstacles)
        
        return {
            'success': True,
            'collision': collision,
            'path': path,
            'path_length': path_length,
            'min_clearance': min_clearance,
            'planning_time': planning_time,
            'tau': tau
        }
    
    def check_collision_grid(self, path, grid: np.ndarray, 
                            origin: tuple, resolution: float) -> bool:
        """Check if path collides with obstacles in grid"""
        if path is None:
            return False
        
        # Convert to numpy array if it's a list of tuples
        if isinstance(path, list):
            path = np.array(path)
        
        robot_radius_pixels = int(np.ceil(self.robot_radius / resolution))
        
        for point in path:
            x, y = point
            
            # Convert to grid coordinates
            grid_x = int((x - origin[0]) / resolution)
            grid_y = int((y - origin[1]) / resolution)
            
            # Check robot footprint
            for di in range(-robot_radius_pixels, robot_radius_pixels + 1):
                for dj in range(-robot_radius_pixels, robot_radius_pixels + 1):
                    check_x = grid_x + dj
                    check_y = grid_y + di
                    
                    # Check bounds
                    if (0 <= check_x < grid.shape[1] and 
                        0 <= check_y < grid.shape[0]):
                        
                        # Check if within robot radius
                        if di*di + dj*dj <= robot_radius_pixels*robot_radius_pixels:
                            if grid[check_y, check_x] == 100:  # Occupied
                                return True  # Collision detected
        
        return False
    
    def point_to_rectangle_distance(self, point: np.ndarray, rect: np.ndarray) -> float:
        """Distance from point to rectangle"""
        x, y = point
        x1, y1, x2, y2 = rect
        
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 0.0
        
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return np.sqrt(dx*dx + dy*dy)
    
    def compute_path_length(self, path) -> float:
        """Compute total path length"""
        if path is None or len(path) < 2:
            return 0.0
        
        # Convert to numpy array if it's a list of tuples
        if isinstance(path, list):
            path = np.array(path)
        
        length = 0.0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        return length
    
    def compute_min_clearance(self, path, obstacles: np.ndarray) -> float:
        """Compute minimum clearance along path"""
        if path is None:
            return 0.0
        
        # Convert to numpy array if it's a list of tuples
        if isinstance(path, list):
            path = np.array(path)
        
        min_clearance = float('inf')
        for point in path:
            for obs in obstacles:
                dist = self.point_to_rectangle_distance(point, obs)
                min_clearance = min(min_clearance, dist)
        
        return min_clearance
    
    def create_occupancy_grid(self, obstacles: np.ndarray, resolution: float = 0.1) -> np.ndarray:
        """Create occupancy grid from obstacles (for noise model)"""
        # Simple grid creation - this is a placeholder
        # In practice, should match the MRPB grid format
        grid_size = 100
        grid = np.zeros((grid_size, grid_size))
        
        # Mark obstacles in grid
        for obs in obstacles:
            # Convert to grid coordinates (simplified)
            x1, y1, x2, y2 = obs
            # ... grid marking logic ...
        
        return grid
    
    def extract_obstacles_from_grid(self, grid: np.ndarray, origin: tuple, resolution: float) -> np.ndarray:
        """Extract obstacles from occupancy grid as bounding boxes"""
        obstacles = []
        
        # Find connected components of occupied cells
        occupied = (grid == 100).astype(np.uint8)
        
        # Use simple approach: treat each occupied cell as a small obstacle
        # This is sufficient for feature extraction
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if occupied[y, x]:
                    # Convert to world coordinates
                    world_x = origin[0] + x * resolution
                    world_y = origin[1] + y * resolution
                    # Store as [x1, y1, x2, y2] format
                    obstacles.append([world_x, world_y, 
                                    world_x + resolution, world_y + resolution])
        
        return np.array(obstacles) if obstacles else np.array([])
    
    def compute_statistics(self, results: Dict) -> Dict:
        """Compute statistics for each method"""
        stats = {}
        
        for method in ['naive', 'standard_cp', 'learnable_cp']:
            method_results = results[method]
            
            # Success rate
            successes = [r['success'] for r in method_results]
            success_rate = np.mean(successes) if successes else 0.0
            
            # Collision rate
            collisions = [r['collision'] for r in method_results if r['success']]
            collision_rate = np.mean(collisions) if collisions else 0.0
            
            # Path length
            lengths = [r['path_length'] for r in method_results if r['success'] and not r['collision']]
            avg_length = np.mean(lengths) if lengths else 0.0
            
            # Min clearance
            clearances = [r['min_clearance'] for r in method_results if r['success']]
            avg_clearance = np.mean(clearances) if clearances else 0.0
            
            # Planning time
            times = [r['planning_time'] for r in method_results]
            avg_time = np.mean(times) if times else 0.0
            
            stats[method] = {
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'avg_path_length': avg_length,
                'avg_min_clearance': avg_clearance,
                'avg_planning_time': avg_time,
                'num_trials': len(method_results)
            }
            
            # Additional stats for learnable CP
            if method == 'learnable_cp':
                tau_means = [r['tau_mean'] for r in method_results if r['success'] and 'tau_mean' in r]
                stats[method]['avg_tau'] = np.mean(tau_means) if tau_means else 0.0
                stats[method]['tau_std'] = np.std(tau_means) if tau_means else 0.0
        
        return stats
    
    def save_results(self, results: Dict, statistics: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results
        results_path = self.results_dir / f'results_{timestamp}.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for method, method_results in results.items():
                json_results[method] = []
                for r in method_results:
                    json_r = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in r.items()}
                    json_results[method].append(json_r)
            json.dump(json_results, f, indent=2)
        
        # Save statistics
        stats_path = self.results_dir / f'statistics_{timestamp}.json'
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        
        # Save summary
        summary_path = self.results_dir / f'summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("LEARNABLE CP EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for method, stats in statistics.items():
                f.write(f"{method.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Success Rate: {stats['success_rate']:.1%}\n")
                f.write(f"Collision Rate: {stats['collision_rate']:.1%}\n")
                f.write(f"Avg Path Length: {stats['avg_path_length']:.2f}m\n")
                f.write(f"Avg Min Clearance: {stats['avg_min_clearance']:.3f}m\n")
                f.write(f"Avg Planning Time: {stats['avg_planning_time']:.3f}s\n")
                
                if method == 'learnable_cp' and 'avg_tau' in stats:
                    f.write(f"Avg Tau: {stats['avg_tau']:.3f}m ± {stats['tau_std']:.3f}m\n")
                
                f.write("\n")
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def generate_comparison_plots(self, statistics: Dict):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        methods = list(statistics.keys())
        colors = ['red', 'blue', 'green']
        
        # Success rate
        ax = axes[0, 0]
        success_rates = [statistics[m]['success_rate'] * 100 for m in methods]
        ax.bar(methods, success_rates, color=colors)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Planning Success Rate')
        ax.set_ylim([0, 105])
        for i, v in enumerate(success_rates):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')
        
        # Collision rate
        ax = axes[0, 1]
        collision_rates = [statistics[m]['collision_rate'] * 100 for m in methods]
        ax.bar(methods, collision_rates, color=colors)
        ax.set_ylabel('Collision Rate (%)')
        ax.set_title('Collision Rate (Lower is Better)')
        for i, v in enumerate(collision_rates):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        # Path length
        ax = axes[0, 2]
        path_lengths = [statistics[m]['avg_path_length'] for m in methods]
        ax.bar(methods, path_lengths, color=colors)
        ax.set_ylabel('Path Length (m)')
        ax.set_title('Average Path Length')
        for i, v in enumerate(path_lengths):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center')
        
        # Min clearance
        ax = axes[1, 0]
        clearances = [statistics[m]['avg_min_clearance'] for m in methods]
        ax.bar(methods, clearances, color=colors)
        ax.set_ylabel('Min Clearance (m)')
        ax.set_title('Average Minimum Clearance')
        for i, v in enumerate(clearances):
            ax.text(i, v + 0.001, f'{v:.3f}', ha='center')
        
        # Planning time
        ax = axes[1, 1]
        times = [statistics[m]['avg_planning_time'] for m in methods]
        ax.bar(methods, times, color=colors)
        ax.set_ylabel('Planning Time (s)')
        ax.set_title('Average Planning Time')
        for i, v in enumerate(times):
            ax.text(i, v + 0.001, f'{v:.3f}', ha='center')
        
        # Combined score (custom metric)
        ax = axes[1, 2]
        # Score = success_rate * (1 - collision_rate) / (1 + normalized_path_length)
        scores = []
        for m in methods:
            s = statistics[m]
            score = (s['success_rate'] * (1 - s['collision_rate']) / 
                    (1 + s['avg_path_length'] / 50.0))  # Normalize by 50m
            scores.append(score)
        
        ax.bar(methods, scores, color=colors)
        ax.set_ylabel('Combined Score')
        ax.set_title('Overall Performance Score')
        for i, v in enumerate(scores):
            ax.text(i, v + 0.001, f'{v:.3f}', ha='center')
        
        plt.suptitle('Learnable CP vs Baselines Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Saved comparison plot to {plot_path}")


def main():
    """Main function to train and evaluate learnable CP"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Learnable CP for Path Planning')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], 
                       default='evaluate', help='Mode to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='learn_cp_config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.mode in ['train', 'both']:
        logging.info("Starting training...")
        trainer = LearnableCPTrainer(args.config)
        trainer.train()
        logging.info("Training complete!")
    
    if args.mode in ['evaluate', 'both']:
        logging.info("Starting evaluation...")
        evaluator = LearnableCPEvaluator(args.config)
        
        # Load trained model
        evaluator.load_trained_model(args.checkpoint)
        
        # Run evaluation
        statistics = evaluator.evaluate_all_methods()
        
        # Print summary
        logging.info("\n" + "="*50)
        logging.info("EVALUATION SUMMARY")
        logging.info("="*50)
        
        for method, stats in statistics.items():
            logging.info(f"\n{method.upper()}:")
            logging.info(f"  Success Rate: {stats['success_rate']:.1%}")
            logging.info(f"  Collision Rate: {stats['collision_rate']:.1%}")
            logging.info(f"  Avg Path Length: {stats['avg_path_length']:.2f}m")
            logging.info(f"  Avg Min Clearance: {stats['avg_min_clearance']:.3f}m")
            
            if method == 'learnable_cp' and 'avg_tau' in stats:
                logging.info(f"  Adaptive Tau: {stats['avg_tau']:.3f}m ± {stats['tau_std']:.3f}m")
        
        logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()