#!/usr/bin/env python3
"""
Run naive method with noise for comparison with Standard CP
Uses same noise types and metrics as Standard CP ablation study
"""

import sys
import os
sys.path.append(os.path.abspath('../..'))
sys.path.append('/mnt/ssd1/divake/path_planning/uncertainty_path_planning')
sys.path.append('/mnt/ssd1/divake/path_planning/uncertainty_path_planning/continuous_planner')

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import yaml
import time
from multiprocessing import Pool, cpu_count

# Import from Standard CP for consistency
sys.path.append('../standard_cp/ablation_studies')
from noise_model import NoiseModel
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid

# Import metrics
from mrpb_metrics import MRPBMetrics, NavigationData, calculate_obstacle_distance

def calculate_path_metrics(path, occupancy_grid, origin, resolution, robot_radius):
    """
    Calculate path metrics (d0, davg, p0) for a given path on the occupancy grid
    Same as Standard CP for consistency
    """
    if not path or len(path) < 2:
        return {'d_0': 0, 'd_avg': 0, 'p_0': 0}
    
    # Calculate distances to obstacles for each waypoint
    distances = []
    danger_count = 0
    safe_distance = 0.3  # meters
    
    for point in path:
        # Calculate minimum distance to obstacles
        min_dist = calculate_obstacle_distance(point, occupancy_grid, origin, resolution)
        distances.append(min_dist)
        
        if min_dist < safe_distance:
            danger_count += 1
    
    # Calculate metrics
    d_0 = min(distances) if distances else 0  # Initial clearance (minimum)
    d_avg = np.mean(distances) if distances else 0  # Average clearance
    p_0 = (danger_count / len(path)) * 100 if path else 0  # Percentage in danger zone
    
    return {
        'd_0': d_0,
        'd_avg': d_avg,
        'p_0': p_0
    }

def run_single_trial(args):
    """
    Run a single trial for parallelization
    Returns: (path_length, metrics, planning_time) or None if failed
    """
    trial, start, goal, noise_types, noise_level, clean_grid, map_info, robot_radius, params = args
    
    # Initialize noise model for this worker
    noise_model = NoiseModel('/mnt/ssd1/divake/path_planning/uncertainty_path_planning/continuous_planner/config/standard_cp_config.yaml')
    noise_model.noise_config['noise_types'] = noise_types
    
    # Add noise with unique seed (same seed pattern as Standard CP)
    test_id = map_info.get('test_id', 1)
    noisy_grid = noise_model.add_realistic_noise(
        clean_grid, noise_level, seed=trial + test_id*1000
    )
    
    # Plan path with timing (using naive RRT* - same as test_runner.py)
    planning_start = time.time()
    
    planner = RRTStarGrid(
        start=start,
        goal=goal,
        occupancy_grid=noisy_grid,
        origin=map_info['origin'],
        resolution=map_info['resolution'],
        robot_radius=robot_radius,
        step_size=params.get('step_size', 0.8),
        max_iter=params.get('max_iterations', 15000),
        goal_threshold=params.get('goal_threshold', 0.5),
        search_radius=params.get('search_radius', 2.5),
        early_termination=True,
        seed=trial
    )
    
    path = planner.plan()
    planning_time = time.time() - planning_start
    
    if path is not None and len(path) > 0:
        # Calculate path length
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                        for i in range(len(path)-1))
        
        # Calculate metrics on noisy map (same as Standard CP)
        metrics = calculate_path_metrics(
            path, noisy_grid, map_info['origin'], 
            map_info['resolution'], robot_radius
        )
        metrics['waypoints'] = len(path)
        
        return path_length, metrics, planning_time
    
    return None

def run_naive_ablation():
    """
    Run naive method with noise for comparison with Standard CP
    """
    print("="*80)
    print("NAIVE METHOD WITH NOISE - FOR COMPARISON WITH STANDARD CP")
    print("="*80)
    print("Testing noise types (same as Standard CP):")
    print("1. Transparency noise - Glass/transparent obstacles")
    print("2. Occlusion noise - Partial visibility")
    print("3. Localization drift - Position uncertainty")
    print("4. Combined - All noise types together")
    print("="*80)
    
    # Configuration (same as Standard CP)
    num_trials = 5  # Testing with 1, change to 5 for full run
    noise_level = 0.25  # Same as Standard CP
    
    # Load configs
    config_dir = '../../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, 'config_algorithm.yaml'), 'r') as f:
        algo_config = yaml.safe_load(f)
    
    # Get test environments
    test_environments = algo_config['experiments']['test_environments']
    
    print(f"\nTesting {len(test_environments)} environments: {', '.join(test_environments)}")
    print(f"Trials: {num_trials} per noise type per test ID")
    print(f"Noise level: {noise_level}")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    
    # Test each noise type (same as Standard CP)
    noise_types_to_test = [
        (['transparency_noise'], 'transparency'),
        (['occlusion_noise'], 'occlusion'),
        (['localization_drift'], 'localization'),
        (['transparency_noise', 'occlusion_noise', 'localization_drift'], 'combined')
    ]
    
    # Store results
    all_results = {}
    
    # Process each environment
    for test_env in test_environments:
        print(f"\n{'='*70}")
        print(f"TESTING ENVIRONMENT: {test_env}")
        print(f"{'='*70}")
        
        # Skip environments without tests
        if test_env not in env_config['environments']:
            print(f"Skipping {test_env}: Not in config")
            continue
            
        env_tests = env_config['environments'][test_env].get('tests', [])
        if not env_tests:
            print(f"Skipping {test_env}: No tests configured")
            continue
        
        # Load map once for this environment
        parser = MRPBMapParser(
            map_name=test_env,
            mrpb_path='../../mrpb_dataset/'
        )
        clean_grid = parser.occupancy_grid.copy()
        
        # Get environment-specific parameters
        params = algo_config['planning']['rrt_star'].copy()
        if 'env_overrides' in algo_config['planning']:
            if test_env in algo_config['planning']['env_overrides']:
                overrides = algo_config['planning']['env_overrides'][test_env]
                params.update(overrides)
        
        # Map info for passing to workers
        map_info = {
            'origin': parser.origin,
            'resolution': parser.resolution,
            'test_id': 1  # Will be updated for each test
        }
        
        robot_radius = env_config['robot']['radius']
        
        # Test ALL test IDs for this environment
        for test in env_tests:
            test_id = test.get('id', 1)
            start = test['start']
            goal = test['goal']
            map_info['test_id'] = test_id
            
            print(f"\nTesting {test_env} - Test ID {test_id}")
            print(f"Start: {start}, Goal: {goal}")
            
            # Store results for this test
            test_key = f"{test_env}_test{test_id}"
            all_results[test_key] = {}
            
            for noise_types, label in noise_types_to_test:
                print(f"\n{'='*50}")
                print(f"Testing: {label} noise")
                print(f"{'='*50}")
                
                # Prepare arguments for parallel processing
                trial_args = []
                for trial in range(num_trials):
                    args = (trial, start, goal, noise_types, noise_level, 
                           clean_grid, map_info, robot_radius, params)
                    trial_args.append(args)
                
                # Run trials in parallel
                print(f"Running {num_trials} trials in parallel...")
                
                path_lengths = []
                all_d0 = []
                all_davg = []
                all_p0 = []
                all_waypoints = []
                all_planning_times = []
                planning_failures = 0
                
                # Use multiprocessing Pool
                with Pool(processes=min(num_trials, cpu_count())) as pool:
                    results = list(tqdm(
                        pool.imap(run_single_trial, trial_args),
                        total=num_trials,
                        desc=f"{label}-{test_env}-t{test_id}"
                    ))
                
                # Process results
                for result in results:
                    if result is not None:
                        path_length, metrics, planning_time = result
                        path_lengths.append(path_length)
                        all_d0.append(metrics['d_0'])
                        all_davg.append(metrics['d_avg'])
                        all_p0.append(metrics['p_0'])
                        all_waypoints.append(metrics['waypoints'])
                        all_planning_times.append(planning_time)
                    else:
                        planning_failures += 1
                
                # Store aggregated metrics
                success_rate = (num_trials - planning_failures) / num_trials
                
                if path_lengths:  # If at least one success
                    result_data = {
                        'success_rate': success_rate,
                        'planning_failures': planning_failures,
                        'path_length': {
                            'mean': float(np.mean(path_lengths)),
                            'std': float(np.std(path_lengths))
                        },
                        'd_0': {
                            'mean': float(np.mean(all_d0)),
                            'std': float(np.std(all_d0))
                        },
                        'd_avg': {
                            'mean': float(np.mean(all_davg)),
                            'std': float(np.std(all_davg))
                        },
                        'p_0': {
                            'mean': float(np.mean(all_p0)),
                            'std': float(np.std(all_p0))
                        },
                        'waypoints': {
                            'mean': float(np.mean(all_waypoints)),
                            'std': float(np.std(all_waypoints))
                        },
                        'time': {
                            'mean': float(np.mean(all_planning_times)),
                            'std': float(np.std(all_planning_times))
                        }
                    }
                else:
                    result_data = {
                        'success_rate': 0.0,
                        'planning_failures': planning_failures,
                        'error': 'All trials failed'
                    }
                
                all_results[test_key][label] = result_data
                
                # Print summary
                print(f"\nResults for {label}:")
                print(f"  Success rate: {success_rate*100:.1f}%")
                if path_lengths:
                    print(f"  Path length: {result_data['path_length']['mean']:.2f} ± {result_data['path_length']['std']:.2f} m")
                    print(f"  d₀: {result_data['d_0']['mean']:.3f} ± {result_data['d_0']['std']:.3f} m")
                    print(f"  d_avg: {result_data['d_avg']['mean']:.3f} ± {result_data['d_avg']['std']:.3f} m")
                    print(f"  p₀: {result_data['p_0']['mean']:.2f} ± {result_data['p_0']['std']:.2f} %")
                    print(f"  Time: {result_data['time']['mean']:.2f} ± {result_data['time']['std']:.2f} s")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results/naive_with_noise')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = results_dir / f'naive_noise_ablation_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*80}")
    print(f"Results saved to {json_path}")
    
    # Create metrics table
    create_metrics_table(all_results, results_dir, timestamp)
    
    # Print final summary
    print_summary(all_results)

def create_metrics_table(all_results, results_dir, timestamp):
    """
    Create a visual table with metrics (same format as Standard CP)
    """
    table_data = []
    
    for test_key, test_results in all_results.items():
        env_name = test_key.rsplit('_test', 1)[0]
        test_id = test_key.rsplit('_test', 1)[1]
        
        for noise_type, data in test_results.items():
            if 'path_length' in data:  # Only if successful
                def format_metric(metric_dict):
                    if isinstance(metric_dict, dict) and 'mean' in metric_dict:
                        mean = metric_dict['mean']
                        std = metric_dict.get('std', 0)
                        if std > 0.01:
                            return f"{mean:.2f}±{std:.2f}"
                        else:
                            return f"{mean:.2f}"
                    return "N/A"
                
                row_data = [
                    f"{env_name}-{test_id}",
                    noise_type.capitalize(),
                    f"{data['success_rate']*100:.1f}%",
                    format_metric(data.get('path_length', {})),
                    format_metric(data.get('waypoints', {})),
                    format_metric(data.get('d_0', {})),
                    format_metric(data.get('d_avg', {})),
                    format_metric(data.get('p_0', {})),
                    format_metric(data.get('time', {}))
                ]
                table_data.append(row_data)
    
    if not table_data:
        print("No metrics data available")
        return
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(18, max(8, len(table_data)*0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Column headers
    col_labels = ['Test', 'Noise Type', 'Success', 'Path Length (m)', 
                  'Waypoints', 'd₀ (m)', 'd_avg (m)', 'p₀ (%)', 'T (s)']
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12, 0.11, 0.09, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color headers
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('NAIVE Method Metrics with Different Noise Types', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Save the table
    table_path = results_dir / f'naive_metrics_table_{timestamp}.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Metrics table saved to {table_path}")
    
    # Also save as CSV
    csv_data = pd.DataFrame(table_data, columns=col_labels)
    csv_path = results_dir / f'naive_metrics_table_{timestamp}.csv'
    csv_data.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

def print_summary(all_results):
    """
    Print aggregated summary across all tests
    """
    print("\n" + "="*80)
    print("AGGREGATED SUMMARY ACROSS ALL TESTS")
    print("="*80)
    
    noise_types = ['transparency', 'occlusion', 'localization', 'combined']
    
    for noise_type in noise_types:
        all_success_rates = []
        all_path_lengths = []
        all_d0 = []
        all_davg = []
        all_p0 = []
        
        for test_results in all_results.values():
            if noise_type in test_results:
                data = test_results[noise_type]
                all_success_rates.append(data['success_rate'])
                if 'path_length' in data:
                    all_path_lengths.append(data['path_length']['mean'])
                    all_d0.append(data['d_0']['mean'])
                    all_davg.append(data['d_avg']['mean'])
                    all_p0.append(data['p_0']['mean'])
        
        if all_success_rates:
            print(f"\n{noise_type.upper()}:")
            print(f"  Average success rate: {np.mean(all_success_rates)*100:.1f}%")
            if all_path_lengths:
                print(f"  Average path length: {np.mean(all_path_lengths):.2f} m")
                print(f"  Average d₀: {np.mean(all_d0):.3f} m")
                print(f"  Average d_avg: {np.mean(all_davg):.3f} m")
                print(f"  Average p₀: {np.mean(all_p0):.2f}%")

if __name__ == "__main__":
    run_naive_ablation()