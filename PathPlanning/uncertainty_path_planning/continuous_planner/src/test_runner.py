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
from multiprocessing import Pool, cpu_count
from functools import partial

from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
from mrpb_metrics import MRPBMetrics, NavigationData


def run_single_test_worker(config, test_case):
    """Worker function for parallel test execution
    
    Args:
        config: Configuration dictionary
        test_case: Tuple of (env_name, test_id)
    
    Returns:
        Dictionary with test results
    """
    env_name, test_id = test_case
    print(f"[Worker] Starting {env_name} - Test {test_id}")
    
    # Get environment and test configuration
    env_config = config['environments'][env_name]
    test_config = None
    for test in env_config['tests']:
        if test['id'] == test_id:
            test_config = test
            break
    
    if not test_config:
        return {'naive': {'success': False, 'error': 'Test config not found'}}
    
    # Parse MRPB map
    parser = MRPBMapParser(env_name, '../mrpb_dataset')
    
    # Get start and goal
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])
    
    # Initialize metrics
    metrics = MRPBMetrics(safe_distance=0.3)
    
    # Get parameters with environment-specific overrides  
    params = config['planning']['rrt_star'].copy()
    if 'env_overrides' in config['planning']:
        if env_name in config['planning']['env_overrides']:
            overrides = config['planning']['env_overrides'][env_name]
            params.update(overrides)
    
    # Run RRT* planner
    planner = RRTStarGrid(
        start=start,
        goal=goal,
        occupancy_grid=parser.occupancy_grid,
        origin=parser.origin,
        resolution=parser.resolution,
        robot_radius=config['robot']['radius'],
        step_size=params['step_size'],
        max_iter=params['max_iterations'],
        goal_threshold=params['goal_threshold'],
        search_radius=params['search_radius']
    )
    
    start_time = time.time()
    path = planner.plan()
    planning_time = time.time() - start_time
    
    result = {
        'path': path,
        'planning_time': planning_time,
        'success': path is not None
    }
    
    if path:
        path_length = planner.compute_path_length(path)
        result['path_length'] = path_length
        
        # Calculate metrics (simplified for parallel execution)
        timestamp = 0.0
        dt = 0.1
        
        for i, point in enumerate(path):
            # Simplified obstacle distance calculation
            obs_dist = 0.5  # Placeholder
            v = 0.5 if i > 0 else 0.0
            
            data = NavigationData(
                timestamp=timestamp,
                x=point[0],
                y=point[1],
                theta=0.0,
                v=v,
                omega=0.0,
                obs_dist=obs_dist,
                time_cost=planning_time / len(path)
            )
            metrics.log_data(data)
            timestamp += dt
        
        result['metrics'] = metrics.compute_all_metrics()
        
        print(f"[Worker] SUCCESS {env_name}-{test_id}: {path_length:.2f}m in {planning_time:.1f}s")
    else:
        result['path_length'] = -1
        print(f"[Worker] FAILED {env_name}-{test_id} after {planning_time:.1f}s")
    
    return {'naive': result}


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
        print(f"Method: Naive (no uncertainty consideration)")
    
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
        
        # Test naive method only
        print(f"\n--- Testing NAIVE ---")
        results['naive'] = self.test_naive(parser, start, goal, env_name)
        
        return results
    
    def test_naive(self, parser: MRPBMapParser, start: Tuple, goal: Tuple, env_name: str = None) -> Dict:
        """Test naive planner (no uncertainty) with MRPB metrics"""
        
        # Initialize metrics calculator
        metrics = MRPBMetrics(safe_distance=0.3)  # 30cm safety distance
        
        # Get parameters with environment-specific overrides
        params = self.planning_params['rrt_star'].copy()
        if env_name and 'env_overrides' in self.planning_params:
            if env_name in self.planning_params['env_overrides']:
                overrides = self.planning_params['env_overrides'][env_name]
                params.update(overrides)
                print(f"  Using environment-specific parameters for {env_name}")
        
        # Run RRT* with occupancy grid
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=parser.occupancy_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=self.robot_radius,
            step_size=params['step_size'],
            max_iter=params['max_iterations'],
            goal_threshold=params['goal_threshold'],
            search_radius=params['search_radius']
        )
        
        start_time = time.time()
        path = planner.plan()
        planning_time = time.time() - start_time
        
        if path:
            path_length = planner.compute_path_length(path)
            print(f"  Path found: {len(path)} waypoints, {path_length:.2f} m")
            
            # Simulate navigation and collect metrics
            # For now, we'll create synthetic navigation data from the path
            timestamp = 0.0
            dt = 0.1  # 100ms between waypoints
            
            for i, point in enumerate(path):
                # Calculate distance to nearest obstacle
                obs_dist = self.calculate_obstacle_distance(point, parser.occupancy_grid, 
                                                            parser.origin, parser.resolution)
                
                # Calculate velocity (simplified)
                if i > 0:
                    dx = path[i][0] - path[i-1][0]
                    dy = path[i][1] - path[i-1][1]
                    dist = np.sqrt(dx**2 + dy**2)
                    v = dist / dt
                else:
                    v = 0.0
                
                # Log navigation data
                data = NavigationData(
                    timestamp=timestamp,
                    x=point[0],
                    y=point[1],
                    theta=0.0,  # Simplified - not tracking orientation
                    v=v,
                    omega=0.0,  # Simplified - not tracking angular velocity
                    obs_dist=obs_dist,
                    time_cost=planning_time / len(path)  # Distribute planning time
                )
                metrics.log_data(data)
                timestamp += dt
            
            # Compute all metrics
            all_metrics = metrics.compute_all_metrics()
            
            # Print metrics summary
            print(f"\n  METRICS SUMMARY:")
            print(f"    Safety: d_0={all_metrics['safety']['d_0']:.3f}m, "
                  f"p_0={all_metrics['safety']['p_0']:.1f}%")
            print(f"    Efficiency: T={all_metrics['efficiency']['T']:.2f}s, "
                  f"C={all_metrics['efficiency']['C']:.1f}ms")
            print(f"    Smoothness: f_ps={all_metrics['smoothness']['f_ps']:.4f}m², "
                  f"f_vs={all_metrics['smoothness']['f_vs']:.3f}m/s²")
            
        else:
            print(f"  No path found!")
            path_length = -1
            all_metrics = None
        
        return {
            'path': path,
            'path_length': path_length,
            'planning_time': planning_time,
            'success': path is not None,
            'metrics': all_metrics
        }
    
    def calculate_obstacle_distance(self, point: Tuple[float, float], 
                                   occupancy_grid: np.ndarray,
                                   origin: Tuple[float, float],
                                   resolution: float) -> float:
        """Calculate distance from point to nearest obstacle"""
        # Convert world coordinates to grid indices
        grid_x = int((point[0] - origin[0]) / resolution)
        grid_y = int((point[1] - origin[1]) / resolution)
        
        # Search radius in grid cells
        search_radius = int(2.0 / resolution)  # Search within 2 meters
        
        min_dist = float('inf')
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                # Check bounds
                if (0 <= check_x < occupancy_grid.shape[1] and 
                    0 <= check_y < occupancy_grid.shape[0]):
                    
                    # If occupied cell found
                    if occupancy_grid[check_y, check_x] > 50:
                        # Calculate distance in meters
                        dist = np.sqrt((dx * resolution)**2 + (dy * resolution)**2)
                        min_dist = min(min_dist, dist)
        
        return min_dist
    
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
    
    def run_all_tests(self, parallel=True, num_workers=None):
        """Run all tests specified in configuration
        
        Args:
            parallel: If True, run tests in parallel
            num_workers: Number of parallel workers (default: CPU count)
        """
        
        print("\n" + "="*80)
        print("STARTING FULL EXPERIMENT SUITE")
        if parallel:
            workers = num_workers or cpu_count()
            print(f"Running in PARALLEL with {workers} workers")
        else:
            print("Running in SERIAL mode")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all test cases
        test_cases = []
        for env_name in self.experiment_settings['test_environments']:
            if env_name not in self.environments:
                print(f"Warning: Environment {env_name} not found in config")
                continue
            
            env_config = self.environments[env_name]
            for test in env_config['tests']:
                test_cases.append((env_name, test['id']))
        
        print(f"Total tests to run: {len(test_cases)}")
        
        if parallel:
            # Run tests in parallel
            print(f"Starting parallel execution on {workers} CPU cores...")
            start_time = time.time()
            
            with Pool(processes=workers) as pool:
                # Create a partial function with self bound
                worker_func = partial(run_single_test_worker, self.config)
                
                # Run all tests in parallel
                results = pool.map(worker_func, test_cases)
            
            # Organize results
            for (env_name, test_id), test_result in zip(test_cases, results):
                if env_name not in self.all_results:
                    self.all_results[env_name] = {}
                self.all_results[env_name][f"test_{test_id}"] = test_result
            
            elapsed = time.time() - start_time
            print(f"\nParallel execution completed in {elapsed:.1f} seconds")
            
        else:
            # Original serial execution
            for env_name in self.experiment_settings['test_environments']:
                if env_name not in self.environments:
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
        
        # Save results with metrics
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
        total_tests = 0
        successful_tests = 0
        metrics_summary = []
        
        for env_name, env_results in self.all_results.items():
            print(f"\n{env_name.upper()}:")
            
            for test_name, test_results in env_results.items():
                print(f"  {test_name}:")
                
                # Only naive method results
                if 'naive' in test_results:
                    results = test_results['naive']
                    success = "✓" if results.get('success', False) else "✗"
                    path_length = results.get('path_length', -1)
                    time_taken = results.get('planning_time', -1)
                    
                    print(f"    naive: {success} "
                          f"Length={path_length:6.2f}m "
                          f"Time={time_taken:5.3f}s")
                    
                    total_tests += 1
                    if results.get('success', False):
                        successful_tests += 1
                        if results.get('metrics'):
                            metrics_summary.append({
                                'env': env_name,
                                'test': test_name,
                                'metrics': results['metrics']
                            })
        
        # Print metrics summary for successful tests
        if metrics_summary:
            print("\n" + "="*80)
            print("METRICS SUMMARY FOR SUCCESSFUL TESTS")
            print("="*80)
            print(f"\nSuccess Rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
            print("\n{:<20} {:>10} {:>10} {:>10} {:>10}".format(
                "Environment/Test", "d_0(m)", "p_0(%)", "Length(m)", "f_vs(m/s²)"))
            print("-"*70)
            
            for item in metrics_summary:
                env_test = f"{item['env'][:8]}/{item['test'][5:]}"
                metrics = item['metrics']
                print("{:<20} {:>10.3f} {:>10.1f} {:>10.2f} {:>10.2f}".format(
                    env_test,
                    metrics['safety']['d_0'],
                    metrics['safety']['p_0'],
                    metrics['path_length'],
                    metrics['smoothness']['f_vs']
                ))


def main():
    """Main entry point"""
    
    # Create test runner
    runner = MRPBTestRunner()
    
    # Run all tests in parallel
    runner.run_all_tests(parallel=True)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()