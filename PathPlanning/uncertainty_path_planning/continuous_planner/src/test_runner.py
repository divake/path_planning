#!/usr/bin/env python3
"""
Test Runner for MRPB Uncertainty-Aware Path Planning

This is the MAIN entry point for running experiments.
It reads configuration from config_env.yaml, config_methods.yaml, and config_algorithm.yaml.
"""

import yaml
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from mrpb_map_parser import MRPBMapParser
from rrt_star_planner import RRTStar
from continuous_standard_cp import ContinuousStandardCP
from learnable_cp_final import FinalLearnableCP
from proper_monte_carlo_evaluation import monte_carlo_evaluation_single_env
from continuous_visualization import visualize_comparison


class MRPBTestRunner:
    """Main test runner for MRPB experiments"""
    
    def __init__(self):
        """
        Initialize test runner
        """
        # Load configuration files
        with open('config_env.yaml', 'r') as f:
            env_config = yaml.safe_load(f)
        with open('config_methods.yaml', 'r') as f:
            methods_config = yaml.safe_load(f)
        with open('config_algorithm.yaml', 'r') as f:
            algo_config = yaml.safe_load(f)
        
        # Merge configurations
        self.config = {
            'robot': env_config['robot'],
            'environments': env_config['environments'],
            'planning': algo_config['planning'],
            'noise': methods_config['noise'],
            'evaluation': methods_config['evaluation'],
            'experiments': algo_config['experiments'],
            'methods': methods_config['methods']
        }
        
        # Extract key parameters
        self.robot_radius = self.config['robot']['radius']
        self.planning_params = self.config['planning']
        self.noise_params = self.config['noise']
        self.eval_params = self.config['evaluation']
        self.environments = self.config['environments']
        self.experiment_settings = self.config['experiments']
        
        # Create results directory
        self.results_dir = self.eval_params.get('results_dir', '../results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Results storage
        self.all_results = {}
        
        print("="*80)
        print("MRPB TEST RUNNER INITIALIZED")
        print("="*80)
        print(f"Robot radius: {self.robot_radius} m")
        print(f"Noise level: {self.noise_params['perception_noise_level']*100:.0f}%")
        print(f"Monte Carlo trials: {self.eval_params['num_trials']}")
        print(f"Environments to test: {len(self.experiment_settings['test_environments'])}")
        print(f"Methods: {', '.join(self.experiment_settings['methods'])}")
    
    def run_single_test(self, env_name: str, test_id: int) -> Dict:
        """
        Run a single test configuration
        
        Args:
            env_name: Environment name
            test_id: Test configuration ID
            
        Returns:
            Dictionary with results for all methods
        """
        print(f"\n{'='*60}")
        print(f"Testing: {env_name} - Test {test_id}")
        print(f"{'='*60}")
        
        # Get environment configuration
        env_config = self.environments[env_name]
        test_config = None
        for test in env_config['tests']:
            if test['id'] == test_id:
                test_config = test
                break
        
        if not test_config:
            print(f"Test {test_id} not found for {env_name}")
            return {}
        
        # Parse MRPB map
        print(f"Loading map: {env_name}...")
        parser = MRPBMapParser(env_name, '../mrpb_dataset')
        obstacles = parser.obstacles
        
        # Get start and goal
        start = tuple(test_config['start'])
        goal = tuple(test_config['goal'])
        
        print(f"  Start: {start}")
        print(f"  Goal: {goal}")
        print(f"  Expected distance: {test_config['distance']:.2f} m")
        
        results = {}
        
        # Test each method
        for method in self.experiment_settings['methods_to_test']:
            print(f"\n--- Testing {method.upper()} ---")
            
            if method == 'naive':
                results[method] = self.test_naive(parser, start, goal)
            elif method == 'standard_cp':
                results[method] = self.test_standard_cp(parser, start, goal)
            elif method == 'learnable_cp':
                results[method] = self.test_learnable_cp(parser, start, goal)
        
        return results
    
    def test_naive(self, parser: MRPBMapParser, start: Tuple, goal: Tuple) -> Dict:
        """Test naive planner (no uncertainty)"""
        
        # Run RRT* directly on perceived obstacles
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=parser.obstacles,
            bounds=(parser.origin[0], parser.origin[0] + parser.width_meters,
                   parser.origin[1], parser.origin[1] + parser.height_meters),
            robot_radius=self.robot_radius,
            step_size=self.planning_params['rrt_star']['step_size'],
            max_iter=self.planning_params['rrt_star']['max_iterations'],
            goal_threshold=self.planning_params['rrt_star']['goal_threshold']
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path:
            path_length = planner.compute_path_length(path)
            print(f"  Path found: {len(path)} waypoints, {path_length:.2f} m")
        else:
            print(f"  No path found!")
            path_length = -1
        
        return {
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'success': path is not None
        }
    
    def test_standard_cp(self, parser: MRPBMapParser, start: Tuple, goal: Tuple) -> Dict:
        """Test standard CP planner"""
        
        # TODO: Implement calibration and tau computation
        # For now, using fixed tau
        tau = 0.5  # meters
        
        # Inflate obstacles by tau
        inflated_obstacles = self.inflate_obstacles(parser.obstacles, tau)
        
        # Run RRT* on inflated obstacles
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=inflated_obstacles,
            bounds=(parser.origin[0], parser.origin[0] + parser.width_meters,
                   parser.origin[1], parser.origin[1] + parser.height_meters),
            robot_radius=self.robot_radius,
            step_size=self.planning_params['rrt_star']['step_size'],
            max_iter=self.planning_params['rrt_star']['max_iterations'],
            goal_threshold=self.planning_params['rrt_star']['goal_threshold']
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path:
            path_length = planner.compute_path_length(path)
            print(f"  Path found: {len(path)} waypoints, {path_length:.2f} m")
            print(f"  Tau used: {tau:.3f} m")
        else:
            print(f"  No path found with tau={tau:.3f}!")
            path_length = -1
        
        return {
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'tau': tau,
            'success': path is not None
        }
    
    def test_learnable_cp(self, parser: MRPBMapParser, start: Tuple, goal: Tuple) -> Dict:
        """Test learnable CP planner"""
        
        # TODO: Load trained model and compute adaptive tau
        # For now, using environment-specific tau
        tau_base = 0.3
        tau_adaptive = tau_base * 1.2  # Placeholder for adaptive computation
        
        # Inflate obstacles by adaptive tau
        inflated_obstacles = self.inflate_obstacles(parser.obstacles, tau_adaptive)
        
        # Run RRT* on inflated obstacles
        planner = RRTStar(
            start=start,
            goal=goal,
            obstacles=inflated_obstacles,
            bounds=(parser.origin[0], parser.origin[0] + parser.width_meters,
                   parser.origin[1], parser.origin[1] + parser.height_meters),
            robot_radius=self.robot_radius,
            step_size=self.planning_params['rrt_star']['step_size'],
            max_iter=self.planning_params['rrt_star']['max_iterations'],
            goal_threshold=self.planning_params['rrt_star']['goal_threshold']
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path:
            path_length = planner.compute_path_length(path)
            print(f"  Path found: {len(path)} waypoints, {path_length:.2f} m")
            print(f"  Adaptive tau: {tau_adaptive:.3f} m")
        else:
            print(f"  No path found with tau={tau_adaptive:.3f}!")
            path_length = -1
        
        return {
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'tau': tau_adaptive,
            'success': path is not None
        }
    
    def inflate_obstacles(self, obstacles: List, tau: float) -> List:
        """
        Inflate obstacles by tau for conservative planning
        
        Args:
            obstacles: List of (x, y, width, height) tuples
            tau: Inflation amount in meters
            
        Returns:
            List of inflated obstacles
        """
        inflated = []
        for (x, y, w, h) in obstacles:
            inflated.append((
                x - tau,
                y - tau,
                w + 2*tau,
                h + 2*tau
            ))
        return inflated
    
    def run_all_tests(self):
        """Run all tests specified in configuration"""
        
        print("\n" + "="*80)
        print("STARTING FULL EXPERIMENT SUITE")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for env_name in self.experiment_settings['test_environments']:
            if env_name not in self.environments:
                print(f"Warning: Environment {env_name} not found in config")
                continue
            
            env_config = self.environments[env_name]
            env_results = {}
            
            print(f"\n{'='*70}")
            print(f"ENVIRONMENT: {env_name.upper()}")
            print(f"  Description: {env_config['description']}")
            print(f"  Dimensions: {env_config['dimensions'][0]} x {env_config['dimensions'][1]} m")
            print(f"  Difficulty: {env_config['difficulty']}")
            print(f"  Number of tests: {len(env_config['tests'])}")
            print(f"{'='*70}")
            
            for test in env_config['tests']:
                test_id = test['id']
                test_results = self.run_single_test(env_name, test_id)
                env_results[f"test_{test_id}"] = test_results
            
            self.all_results[env_name] = env_results
        
        # Save results
        results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Generate summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all results"""
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Calculate statistics
        for env_name, env_results in self.all_results.items():
            print(f"\n{env_name.upper()}:")
            
            for test_name, test_results in env_results.items():
                print(f"  {test_name}:")
                
                for method, results in test_results.items():
                    success = "✓" if results.get('success', False) else "✗"
                    path_length = results.get('path_length', -1)
                    time_taken = results.get('planning_time', -1)
                    
                    print(f"    {method:15s}: {success} "
                          f"Length={path_length:6.2f}m "
                          f"Time={time_taken:5.3f}s")


def main():
    """Main entry point"""
    
    # Create test runner
    runner = MRPBTestRunner()
    
    # Run all tests
    runner.run_all_tests()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()