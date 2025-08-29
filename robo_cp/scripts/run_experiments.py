#!/usr/bin/env python3
"""
Main experiment runner for comparing uncertainty methods.
Runs naive, traditional CP, and learnable CP on test environments.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from typing import Dict, List, Tuple, Any
import pandas as pd

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

# Import our modules
from environments.base_environment import UncertaintyEnvironment
from src.feature_extraction import FeatureExtractor
from methods.naive_method import NaiveMethod
from methods.traditional_cp_method import TraditionalCPMethod
from methods.learnable_cp_method import LearnableCPMethod

# Import planner wrapper
from src.planner_wrapper import PlannerWrapper


class ExperimentRunner:
    """Run and compare uncertainty methods."""
    
    def __init__(self, env_size: Tuple[int, int] = (50, 50),
                 num_obstacles: int = 20,
                 num_trials: int = 100,
                 noise_sigma: float = 0.5):
        """
        Initialize experiment runner.
        
        Args:
            env_size: (width, height) of environment
            num_obstacles: Number of obstacles
            num_trials: Number of Monte Carlo trials
            noise_sigma: Standard deviation for obstacle noise
        """
        self.env_size = env_size
        self.num_obstacles = num_obstacles
        self.num_trials = num_trials
        self.noise_sigma = noise_sigma
        
        # Initialize methods
        self.methods = {
            'naive': NaiveMethod(),
            'traditional_cp': TraditionalCPMethod(fixed_margin=1.5),
            'learnable_cp': LearnableCPMethod(min_margin=0.2, max_margin=2.0)
        }
        
        # Results storage
        self.results = {method: [] for method in self.methods.keys()}
        
    def create_test_environment(self) -> UncertaintyEnvironment:
        """Create a challenging test environment."""
        env = UncertaintyEnvironment(width=self.env_size[0], height=self.env_size[1])
        
        # Generate random obstacles
        env.generate_random_obstacles(self.num_obstacles)
        
        # Add some challenging features
        # Add a narrow passage
        env.create_narrow_passage(start_x=20, y=25, length=10, gap=3)
        
        # Add obstacle clusters
        env.add_obstacle_cluster(center=(35, 35), num_obstacles=4)
        env.add_obstacle_cluster(center=(15, 15), num_obstacles=3)
        
        return env
    
    def run_single_trial(self, base_env: UncertaintyEnvironment,
                        method_name: str,
                        start: Tuple[int, int],
                        goal: Tuple[int, int]) -> Dict[str, Any]:
        """
        Run a single trial with noise.
        
        Args:
            base_env: Base environment
            method_name: Name of uncertainty method
            start: Start position
            goal: Goal position
            
        Returns:
            Trial results dictionary
        """
        # Create noisy environment
        env = UncertaintyEnvironment(width=base_env.width, height=base_env.height)
        env.original_obstacles = base_env.add_noise_to_obstacles(self.noise_sigma)
        env.obstacles = env.original_obstacles.copy()
        
        # Apply uncertainty method
        method = self.methods[method_name]
        
        if method_name == 'learnable_cp':
            # Extract features for learnable CP
            feature_extractor = FeatureExtractor(env)
            # For simplicity, use global features
            features = feature_extractor.extract_global_features()
            method.feature_extractor = feature_extractor
        
        method_info = method.apply(env)
        
        # Create planner wrapper
        planner = PlannerWrapper(env)
        
        # Run A* planner
        start_time = time.time()
        
        try:
            # Plan path
            path, cost = planner.plan_astar(start, goal, use_inflated=True)
            
            planning_time = time.time() - start_time
            
            if path:
                # Check collision on TRUE obstacles
                collision = env.check_collision(path, use_original=True, robot_radius=0.5)
                
                # Calculate path length
                path_length = self.calculate_path_length(path)
                
                # Calculate minimum clearance
                min_clearance = min([env.get_clearance(p, use_original=True) for p in path])
                
                success = True
            else:
                collision = False
                path_length = float('inf')
                min_clearance = 0
                success = False
                
        except Exception as e:
            print(f"Planning failed for {method_name}: {e}")
            collision = False
            path_length = float('inf')
            min_clearance = 0
            success = False
            planning_time = time.time() - start_time
            path = []
        
        return {
            'method': method_name,
            'success': success,
            'collision': collision,
            'path_length': path_length,
            'min_clearance': min_clearance,
            'planning_time': planning_time,
            'path': path,
            'margin_info': method_info
        }
    
    def run_monte_carlo(self, env: UncertaintyEnvironment,
                       start: Tuple[int, int],
                       goal: Tuple[int, int]) -> Dict[str, List[Dict]]:
        """
        Run Monte Carlo trials for all methods.
        
        Args:
            env: Base environment
            start: Start position
            goal: Goal position
            
        Returns:
            Results for all methods
        """
        results = {method: [] for method in self.methods.keys()}
        
        print(f"\nRunning {self.num_trials} Monte Carlo trials...")
        
        for trial in range(self.num_trials):
            if trial % 20 == 0:
                print(f"  Trial {trial}/{self.num_trials}")
            
            for method_name in self.methods.keys():
                trial_result = self.run_single_trial(env, method_name, start, goal)
                results[method_name].append(trial_result)
        
        return results
    
    def calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0
        
        length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += np.sqrt(dx**2 + dy**2)
        
        return length
    
    def analyze_results(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Analyze and summarize results.
        
        Args:
            results: Results dictionary
            
        Returns:
            Summary DataFrame
        """
        summary = []
        
        for method_name, method_results in results.items():
            # Calculate metrics
            successes = [r['success'] for r in method_results]
            collisions = [r['collision'] for r in method_results if r['success']]
            path_lengths = [r['path_length'] for r in method_results if r['success']]
            clearances = [r['min_clearance'] for r in method_results if r['success']]
            times = [r['planning_time'] for r in method_results]
            
            summary.append({
                'Method': method_name,
                'Success Rate': np.mean(successes) * 100,
                'Collision Rate': np.mean(collisions) * 100 if collisions else 0,
                'Avg Path Length': np.mean(path_lengths) if path_lengths else 0,
                'Std Path Length': np.std(path_lengths) if path_lengths else 0,
                'Avg Min Clearance': np.mean(clearances) if clearances else 0,
                'Avg Planning Time': np.mean(times),
                'Num Successes': sum(successes),
                'Num Collisions': sum(collisions) if collisions else 0
            })
        
        df = pd.DataFrame(summary)
        return df
    
    def visualize_comparison(self, env: UncertaintyEnvironment,
                            results: Dict[str, List[Dict]],
                            start: Tuple[int, int],
                            goal: Tuple[int, int]):
        """
        Create visualization comparing the three methods.
        
        Args:
            env: Environment
            results: Results dictionary
            start: Start position
            goal: Goal position
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        method_names = ['naive', 'traditional_cp', 'learnable_cp']
        colors = {'naive': 'blue', 'traditional_cp': 'green', 'learnable_cp': 'red'}
        
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            
            # Apply method to environment for visualization
            env_copy = UncertaintyEnvironment(width=env.width, height=env.height)
            env_copy.original_obstacles = env.original_obstacles.copy()
            env_copy.obstacles = env.original_obstacles.copy()
            
            method = self.methods[method_name]
            if method_name == 'learnable_cp':
                feature_extractor = FeatureExtractor(env_copy)
                method.feature_extractor = feature_extractor
            
            method.apply(env_copy)
            
            # Plot environment
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_aspect('equal')
            ax.set_title(f'{method_name.replace("_", " ").title()}')
            
            # Plot original obstacles (solid)
            for obs in env_copy.original_obstacles:
                circle = plt.Circle(obs['center'], obs['radius'], 
                                   color='gray', alpha=0.5)
                ax.add_patch(circle)
            
            # Plot inflated obstacles (dashed)
            if method_name != 'naive':
                for obs in env_copy.obstacles:
                    circle = plt.Circle(obs['center'], obs['radius'],
                                      fill=False, edgecolor='yellow',
                                      linestyle='--', linewidth=2, alpha=0.7)
                    ax.add_patch(circle)
            
            # Plot some successful paths
            method_results = results[method_name]
            num_paths_to_show = 5
            shown = 0
            
            for result in method_results:
                if result['success'] and shown < num_paths_to_show:
                    path = result['path']
                    if path:
                        path = np.array(path)
                        color = 'red' if result['collision'] else colors[method_name]
                        alpha = 0.3 if result['collision'] else 0.6
                        ax.plot(path[:, 0], path[:, 1], color=color, 
                               alpha=alpha, linewidth=1)
                        shown += 1
            
            # Plot start and goal
            ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
            ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
            
            # Add metrics text
            method_summary = next((s for s in self.summary.to_dict('records') 
                                  if s['Method'] == method_name), {})
            
            text = f"Success: {method_summary.get('Success Rate', 0):.1f}%\n"
            text += f"Collision: {method_summary.get('Collision Rate', 0):.1f}%\n"
            text += f"Avg Length: {method_summary.get('Avg Path Length', 0):.1f}"
            
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Uncertainty Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/method_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_full_experiment(self):
        """Run complete experiment pipeline."""
        print("=" * 60)
        print("UNCERTAINTY METHODS COMPARISON EXPERIMENT")
        print("=" * 60)
        
        # Create test environment
        print("\n1. Creating test environment...")
        env = self.create_test_environment()
        print(f"   Environment: {env.width}x{env.height} with {len(env.original_obstacles)} obstacles")
        
        # Define start and goal (adjusted for environment size)
        start = (5, 5)
        goal = (min(45, env.width - 5), min(45, env.height - 5))
        print(f"   Start: {start}, Goal: {goal}")
        
        # Run Monte Carlo trials
        print("\n2. Running Monte Carlo trials...")
        results = self.run_monte_carlo(env, start, goal)
        
        # Analyze results
        print("\n3. Analyzing results...")
        self.summary = self.analyze_results(results)
        
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(self.summary.to_string())
        
        # Save results
        print("\n4. Saving results...")
        os.makedirs('results', exist_ok=True)
        
        # Save summary
        self.summary.to_csv('results/summary.csv', index=False)
        
        # Save detailed results
        with open('results/detailed_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Visualize
        print("\n5. Creating visualizations...")
        self.visualize_comparison(env, results, start, goal)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print("\nKey Findings:")
        
        # Compare methods
        naive_collision = self.summary[self.summary['Method'] == 'naive']['Collision Rate'].values[0]
        trad_collision = self.summary[self.summary['Method'] == 'traditional_cp']['Collision Rate'].values[0]
        learn_collision = self.summary[self.summary['Method'] == 'learnable_cp']['Collision Rate'].values[0]
        
        print(f"  - Naive collision rate: {naive_collision:.1f}%")
        print(f"  - Traditional CP collision rate: {trad_collision:.1f}%")
        print(f"  - Learnable CP collision rate: {learn_collision:.1f}%")
        
        if learn_collision < trad_collision:
            improvement = (trad_collision - learn_collision) / trad_collision * 100
            print(f"  → Learnable CP reduces collisions by {improvement:.1f}% vs Traditional CP")
        
        # Path length comparison
        naive_length = self.summary[self.summary['Method'] == 'naive']['Avg Path Length'].values[0]
        learn_length = self.summary[self.summary['Method'] == 'learnable_cp']['Avg Path Length'].values[0]
        
        if naive_length > 0:
            overhead = (learn_length - naive_length) / naive_length * 100
            print(f"  → Learnable CP path overhead: {overhead:.1f}% vs Naive")


if __name__ == "__main__":
    # Run experiment
    runner = ExperimentRunner(
        env_size=(50, 50),
        num_obstacles=20,
        num_trials=100,
        noise_sigma=0.5
    )
    
    runner.run_full_experiment()