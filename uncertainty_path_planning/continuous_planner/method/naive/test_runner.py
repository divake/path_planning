#!/usr/bin/env python3
"""
Test Runner for MRPB Uncertainty-Aware Path Planning
Clean version with only parallel/single unified implementation
"""

import yaml
import numpy as np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import sys
# Add path for common modules (one level up from method/naive/)
sys.path.append('../..')
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
from mrpb_metrics import MRPBMetrics, NavigationData
from visualization import MRPBVisualizer

# Add path for CurvesGenerator
sys.path.append('/mnt/ssd1/divake/path_planning/CurvesGenerator')
from quintic_polynomial import QuinticPolynomial


def calculate_obstacle_distance_worker(point, occupancy_grid, origin, resolution):
    """Calculate distance to nearest obstacle for worker function"""
    px = int((point[0] - origin[0]) / resolution)
    py = int((point[1] - origin[1]) / resolution)
    
    # Search radius in pixels
    search_radius = 50  # 2.5 meters at 0.05m resolution
    
    min_dist = float('inf')
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            check_x = px + dx
            check_y = py + dy
            
            if (0 <= check_x < occupancy_grid.shape[1] and 
                0 <= check_y < occupancy_grid.shape[0]):
                if occupancy_grid[check_y, check_x] > 50:  # Obstacle
                    dist = np.sqrt(dx**2 + dy**2) * resolution
                    min_dist = min(min_dist, dist)
    
    return min_dist if min_dist < float('inf') else 2.5


def generate_realistic_trajectory(waypoints, parser, planning_time=None, max_vel=1.5, max_acc=2.0):
    """
    Generate realistic robot trajectory from RRT* waypoints using QuinticPolynomial
    
    Args:
        waypoints: List of [x, y] waypoint coordinates
        parser: MRPBMapParser for obstacle distance calculation
        planning_time: Total RRT* planning time to distribute across trajectory
        max_vel: Maximum robot velocity (m/s)
        max_acc: Maximum robot acceleration (m/s²)
    
    Returns:
        List of NavigationData with realistic velocities and timing
    """
    if len(waypoints) < 2:
        return []
    
    trajectory_data = []
    current_time = 0.0
    
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        
        # Calculate segment parameters
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        angle = np.arctan2(y1 - y0, x1 - x0)
        
        # Initial and final velocities (start/stop at endpoints, continuous between)
        v0 = 0.0 if i == 0 else min(max_vel, distance / 0.5)
        v1 = 0.0 if i == len(waypoints) - 2 else min(max_vel, distance / 0.5)
        
        # Initial and final accelerations (zero for smooth motion)
        a0 = 0.0
        a1 = 0.0
        
        # Time for this segment (based on average velocity)
        avg_vel = (v0 + v1) / 2 if (v0 + v1) > 0 else max_vel / 2
        T = max(distance / avg_vel, 0.5)  # Minimum 0.5s per segment
        
        # Generate quintic polynomial trajectories for x and y
        traj_x = QuinticPolynomial(x0, v0 * np.cos(angle), a0, x1, v1 * np.cos(angle), a1, T)
        traj_y = QuinticPolynomial(y0, v0 * np.sin(angle), a0, y1, v1 * np.sin(angle), a1, T)
        
        # Sample trajectory at realistic robot control frequency (50Hz = 0.02s)
        dt = 0.02
        num_steps = max(int(T / dt), 5)  # At least 5 points per segment
        
        for step in range(num_steps):
            t = step * T / num_steps
            timestamp = current_time + t
            
            # Calculate position from quintic polynomial
            x = traj_x.calc_xt(t)
            y = traj_y.calc_xt(t)
            
            # Calculate velocity
            vx = traj_x.calc_dxt(t)
            vy = traj_y.calc_dxt(t)
            v = np.sqrt(vx**2 + vy**2)
            
            # Calculate obstacle distance
            obs_dist = calculate_obstacle_distance_worker([x, y], parser.occupancy_grid, 
                                                        parser.origin, parser.resolution)
            
            # Calculate actual computation time per waypoint
            if planning_time is not None and len(waypoints) > 0:
                # Distribute planning time across trajectory points
                # Total trajectory points = sum of num_steps for all segments
                total_trajectory_points = (len(waypoints) - 1) * num_steps
                time_cost = planning_time / total_trajectory_points if total_trajectory_points > 0 else 0.01
            else:
                time_cost = 0.01  # Default fallback
            
            # Create navigation data point
            nav_data = NavigationData(
                timestamp=timestamp,
                x=x,
                y=y,
                theta=np.arctan2(vy, vx),
                v=v,
                omega=0.0,  # Simplified for now
                obs_dist=obs_dist,
                time_cost=time_cost  # Use actual computation time
            )
            
            trajectory_data.append(nav_data)
        
        current_time += T
    
    return trajectory_data


def run_single_test_worker(config, test_case):
    """Worker function for test execution (handles both single and parallel)
    
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
    # Use absolute path for dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', '..', 'mrpb_dataset')
    parser = MRPBMapParser(env_name, dataset_path)
    
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
    
    # Run RRT* planner with early termination for fast path finding
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
        search_radius=params['search_radius'],
        early_termination=True  # Stop at first valid path (no optimization)
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
        result['num_waypoints'] = len(path)
        
        # Generate realistic trajectory from RRT* waypoints using QuinticPolynomial
        # Pass planning_time to calculate actual computation cost per cycle
        trajectory_data = generate_realistic_trajectory(path, parser, planning_time)
        
        # Log trajectory data for metrics calculation
        for nav_data in trajectory_data:
            metrics.log_data(nav_data)
        
        result['metrics'] = metrics.compute_all_metrics()
        
        print(f"[Worker] SUCCESS {env_name}-{test_id}: {path_length:.2f}m in {planning_time:.1f}s")
    else:
        print(f"[Worker] FAILED {env_name}-{test_id} after {planning_time:.1f}s")
        result['path_length'] = -1
        result['num_waypoints'] = 0
        result['metrics'] = {}
    
    return {'naive': result}


class MRPBTestRunner:
    """Main test runner for MRPB experiments - Clean unified version"""
    
    def __init__(self):
        """Initialize test runner"""
        # Get the absolute path to config files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(script_dir, '..', '..', 'config')
        
        # Load configuration files
        config_env_path = os.path.join(config_dir, 'config_env.yaml')
        config_methods_path = os.path.join(config_dir, 'config_methods.yaml')
        config_algo_path = os.path.join(config_dir, 'config_algorithm.yaml')
        
        print(f"Loading configs from: {config_dir}")
        
        with open(config_env_path, 'r') as f:
            env_config = yaml.safe_load(f)
        with open(config_methods_path, 'r') as f:
            methods_config = yaml.safe_load(f)
        with open(config_algo_path, 'r') as f:
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
        results_base = os.path.join(script_dir, '..', '..', 'results', 'naive')
        self.results_dir = self.eval_params.get('results_dir', results_base)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize visualizer
        plots_dir = os.path.join(self.results_dir, 'plots')
        self.visualizer = MRPBVisualizer(self.results_dir, plots_dir)
        
        # Results storage
        self.all_results = {}
        
        # Print configuration summary
        print("="*80)
        print("MRPB TEST RUNNER INITIALIZED")
        print("="*80)
        print(f"Robot radius: {self.robot_radius} m")
        print(f"Noise level: {int(self.noise_params['perception_noise_level']*100)}%")
        print(f"Monte Carlo trials: {self.eval_params['num_trials']}")
        print(f"Environments to test: {len(self.experiment_settings['test_environments'])}")
        print(f"Method: Naive (no uncertainty consideration)")
    
    def run_tests(self, test_cases=None, num_workers=None):
        """
        Run tests with automatic single/parallel handling
        
        Args:
            test_cases: List of (env_name, test_id) tuples. If None, runs all configured tests
            num_workers: Number of parallel workers. If None, auto-detects. If 1, runs serially.
        """
        print("\n" + "="*80)
        print("STARTING EXPERIMENT SUITE")
        
        # Determine number of workers
        if num_workers is None:
            # Use all available CPUs for maximum parallelization
            num_workers = cpu_count()
            print(f"Auto-detected {num_workers} CPUs available")
        
        # Collect test cases if not provided
        if test_cases is None:
            test_cases = []
            for env_name in self.experiment_settings['test_environments']:
                env_config = self.environments[env_name]
                
                # Skip environments without tests (like 'track' which has all tests disabled)
                if 'tests' not in env_config or env_config['tests'] is None:
                    print(f"Skipping {env_name}: No tests configured")
                    continue
                
                for test in env_config['tests']:
                    test_cases.append((env_name, test['id']))
        
        print(f"Total tests to run: {len(test_cases)}")
        
        if len(test_cases) == 0:
            print("No tests to run!")
            return
        
        # Optimize worker count based on test count
        if num_workers > len(test_cases):
            actual_workers = len(test_cases)
            print(f"Optimizing: Using {actual_workers} workers for {len(test_cases)} tests (1 test per worker)")
        else:
            actual_workers = num_workers
            tests_per_worker = len(test_cases) / actual_workers
            print(f"Using {actual_workers} workers (~{tests_per_worker:.1f} tests per worker)")
        
        # Run tests
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if actual_workers == 1 or len(test_cases) == 1:
            # Serial execution
            print("Running in SERIAL mode")
            print("="*80)
            
            for test_case in test_cases:
                result = run_single_test_worker(self.config, test_case)
                env_name, test_id = test_case
                
                if env_name not in self.all_results:
                    self.all_results[env_name] = {}
                self.all_results[env_name][f"test_{test_id}"] = result
        else:
            # Parallel execution
            print(f"Running in PARALLEL mode with {actual_workers} workers")
            print("="*80)
            
            worker_func = partial(run_single_test_worker, self.config)
            
            start_time = time.time()
            with Pool(processes=actual_workers) as pool:
                results = pool.map(worker_func, test_cases)
            
            # Store results
            for (env_name, test_id), result in zip(test_cases, results):
                if env_name not in self.all_results:
                    self.all_results[env_name] = {}
                self.all_results[env_name][f"test_{test_id}"] = result
            
            elapsed = time.time() - start_time
            print(f"\nParallel execution completed in {elapsed:.1f} seconds")
        
        # Save results
        results_file = os.path.join(self.results_dir, f"results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Print summary
        self.print_summary()
        
        # Generate visualizations
        self.visualizer.save_results_to_csv(self.all_results, timestamp)
        self.visualizer.generate_comprehensive_plot(self.all_results, timestamp, self.robot_radius)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*80)
    
    def print_summary(self):
        """Print summary of test results"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for env_name, env_results in self.all_results.items():
            print(f"\n{env_name.upper()}:")
            for test_name, test_result in env_results.items():
                if 'naive' in test_result:
                    result = test_result['naive']
                    if result['success']:
                        print(f"  {test_name}:")
                        print(f"    naive: ✓ Length={result['path_length']:7.2f}m Time={result['planning_time']:.3f}s")
                    else:
                        print(f"  {test_name}:")
                        print(f"    naive: ✗ Length={result['path_length']:7.2f}m Time={result['planning_time']:.3f}s")


def main():
    """Main entry point"""
    runner = MRPBTestRunner()
    
    # Run all configured tests with automatic parallel/serial handling
    runner.run_tests()


if __name__ == "__main__":
    main()