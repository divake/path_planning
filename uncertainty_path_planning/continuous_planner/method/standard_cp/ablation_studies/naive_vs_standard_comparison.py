#!/usr/bin/env python3
"""
Naive vs Standard CP Comparison Study for ICRA 2025
Compares performance metrics between Naive and Standard CP methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Import components
from noise_model import NoiseModel
from nonconformity_scorer import NonconformityScorer
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
from mrpb_metrics import MRPBMetrics


class NaiveVsStandardCPComparison:
    """
    Compare Naive planner (no uncertainty) vs Standard CP (with tau safety margin)
    """
    
    def __init__(self):
        # Load configurations
        self.cp_config_path = '../../../config/standard_cp_config.yaml'
        self.env_config_path = '../../../config/config_env.yaml'
        
        with open(self.cp_config_path, 'r') as f:
            self.cp_config = yaml.safe_load(f)
        
        with open(self.env_config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)
        
        # Initialize components
        self.noise_model = NoiseModel(self.cp_config_path)
        self.nonconformity_scorer = NonconformityScorer(self.cp_config_path)
        self.metrics_calculator = MRPBMetrics()
        
        # Results storage
        self.results = {
            'naive': {},
            'standard_cp': {},
            'comparison': {}
        }
        
        # Create results directory
        self.results_dir = Path('results/ablation_studies/naive_vs_standard')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for environments
        self._env_cache = {}
        
        # Pre-computed tau from calibration (using default or computed value)
        self.tau = None  # Will be computed during calibration phase
    
    def load_environment(self, env_name: str) -> MRPBMapParser:
        """Load and cache environment"""
        if env_name not in self._env_cache:
            self._env_cache[env_name] = MRPBMapParser(
                map_name=env_name,
                mrpb_path='../../../mrpb_dataset'
            )
        return self._env_cache[env_name]
    
    def calibrate_tau(self, num_calibration_trials: int = 100) -> float:
        """
        Calibrate tau using calibration environments
        """
        print("\n" + "="*60)
        print("CALIBRATING TAU FOR STANDARD CP")
        print("="*60)
        
        calibration_envs = self.cp_config['environments']['full_environments']['calibration_envs']
        all_scores = []
        
        trials_per_env = max(1, num_calibration_trials // len(calibration_envs))
        
        with tqdm(total=num_calibration_trials, desc="Calibration") as pbar:
            for env_dict in calibration_envs:
                env_name = env_dict['name']
                env_test_ids = env_dict['test_ids']
                
                parser = self.load_environment(env_name)
                all_env_tests = self.env_config['environments'][env_name]['tests']
                env_tests = [t for t in all_env_tests if t['id'] in env_test_ids]
                
                for trial_idx in range(trials_per_env):
                    test = env_tests[trial_idx % len(env_tests)]
                    seed = trial_idx
                    np.random.seed(seed)
                    
                    # Get grids
                    true_grid = parser.occupancy_grid.copy()
                    noise_level = np.random.choice(self.cp_config['noise_model']['noise_levels'])
                    perceived_grid = self.noise_model.add_realistic_noise(
                        true_grid, noise_level, seed=seed
                    )
                    
                    # Plan path
                    planner = RRTStarGrid(
                        start=test['start'],
                        goal=test['goal'],
                        occupancy_grid=perceived_grid,
                        origin=parser.origin,
                        resolution=parser.resolution,
                        robot_radius=0.17,
                        max_iter=10000,
                        early_termination=True,
                        seed=seed
                    )
                    
                    try:
                        path = planner.plan()
                        if path is not None and len(path) > 0:
                            score = self.nonconformity_scorer.compute_nonconformity_score(
                                true_grid, perceived_grid, path, parser
                            )
                            all_scores.append(score)
                        else:
                            all_scores.append(0.3)  # Max score for failures
                    except:
                        all_scores.append(0.3)
                    
                    pbar.update(1)
        
        # Calculate tau (90th percentile)
        sorted_scores = sorted(all_scores)
        quantile_idx = int(np.ceil((len(sorted_scores) + 1) * 0.9)) - 1
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        self.tau = sorted_scores[quantile_idx]
        
        print(f"\nCalibration complete. Tau = {self.tau:.4f} m")
        print(f"Based on {len(all_scores)} calibration trials")
        
        return self.tau
    
    def run_naive_planner(self, env_name: str, test_config: Dict, 
                         noise_level: float, seed: int) -> Dict:
        """
        Run naive planner (no uncertainty consideration)
        """
        parser = self.load_environment(env_name)
        
        # Get grids
        true_grid = parser.occupancy_grid.copy()
        perceived_grid = self.noise_model.add_realistic_noise(
            true_grid, noise_level, seed=seed
        )
        
        # Plan on perceived grid (naive - no safety margin)
        planner = RRTStarGrid(
            start=test_config['start'],
            goal=test_config['goal'],
            occupancy_grid=perceived_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=0.17,  # Original radius, no inflation
            max_iter=10000,
            early_termination=True,
            seed=seed
        )
        
        result = {
            'method': 'naive',
            'env': env_name,
            'noise_level': noise_level,
            'planning_success': False,
            'collision': False,
            'd_0': None,
            'd_avg': None,
            'path_length': 0
        }
        
        try:
            path = planner.plan()
            
            if path is not None and len(path) > 0:
                result['planning_success'] = True
                result['path_length'] = len(path)
                
                # Check collision on true grid
                collision = self.check_path_collision(path, true_grid, parser)
                result['collision'] = collision
                
                # Calculate metrics
                d_0, d_avg = self.calculate_clearance_metrics(path, true_grid, parser)
                result['d_0'] = d_0
                result['d_avg'] = d_avg
        except:
            pass
        
        return result
    
    def run_standard_cp_planner(self, env_name: str, test_config: Dict, 
                               noise_level: float, seed: int) -> Dict:
        """
        Run Standard CP planner (with tau safety margin)
        """
        parser = self.load_environment(env_name)
        
        # Get grids
        true_grid = parser.occupancy_grid.copy()
        perceived_grid = self.noise_model.add_realistic_noise(
            true_grid, noise_level, seed=seed
        )
        
        # Plan with inflated robot radius (Standard CP)
        inflated_radius = 0.17 + self.tau  # Original radius + safety margin
        
        planner = RRTStarGrid(
            start=test_config['start'],
            goal=test_config['goal'],
            occupancy_grid=perceived_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=inflated_radius,  # Inflated for safety
            max_iter=10000,
            early_termination=True,
            seed=seed
        )
        
        result = {
            'method': 'standard_cp',
            'env': env_name,
            'noise_level': noise_level,
            'tau': self.tau,
            'planning_success': False,
            'collision': False,
            'd_0': None,
            'd_avg': None,
            'path_length': 0
        }
        
        try:
            path = planner.plan()
            
            if path is not None and len(path) > 0:
                result['planning_success'] = True
                result['path_length'] = len(path)
                
                # Check collision on true grid
                collision = self.check_path_collision(path, true_grid, parser)
                result['collision'] = collision
                
                # Calculate metrics
                d_0, d_avg = self.calculate_clearance_metrics(path, true_grid, parser)
                result['d_0'] = d_0
                result['d_avg'] = d_avg
        except:
            pass
        
        return result
    
    def check_path_collision(self, path: List[Tuple[float, float]], 
                            true_grid: np.ndarray, 
                            parser: MRPBMapParser) -> bool:
        """
        Check if path collides with obstacles in true grid
        """
        robot_radius = 0.17
        
        for point in path:
            # Convert to grid coordinates
            grid_x = int((point[0] - parser.origin[0]) / parser.resolution)
            grid_y = int((point[1] - parser.origin[1]) / parser.resolution)
            
            # Check robot footprint
            radius_pixels = int(np.ceil(robot_radius / parser.resolution))
            
            for dx in range(-radius_pixels, radius_pixels + 1):
                for dy in range(-radius_pixels, radius_pixels + 1):
                    px = grid_x + dx
                    py = grid_y + dy
                    
                    if (0 <= px < true_grid.shape[1] and 
                        0 <= py < true_grid.shape[0]):
                        if true_grid[py, px] == 100:  # Occupied
                            return True
        
        return False
    
    def calculate_clearance_metrics(self, path: List[Tuple[float, float]], 
                                   true_grid: np.ndarray,
                                   parser: MRPBMapParser) -> Tuple[float, float]:
        """
        Calculate d_0 (min clearance) and d_avg (average clearance)
        """
        clearances = []
        
        for point in path:
            # Convert to grid coordinates
            grid_x = int((point[0] - parser.origin[0]) / parser.resolution)
            grid_y = int((point[1] - parser.origin[1]) / parser.resolution)
            
            # Find minimum distance to obstacle
            min_dist = float('inf')
            search_radius = 50  # pixels
            
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    px = grid_x + dx
                    py = grid_y + dy
                    
                    if (0 <= px < true_grid.shape[1] and 
                        0 <= py < true_grid.shape[0]):
                        if true_grid[py, px] == 100:  # Occupied
                            dist = np.sqrt(dx**2 + dy**2) * parser.resolution
                            min_dist = min(min_dist, dist)
            
            if min_dist < float('inf'):
                clearances.append(min_dist)
        
        if clearances:
            d_0 = min(clearances)
            d_avg = np.mean(clearances)
        else:
            d_0 = 0.0
            d_avg = 0.0
        
        return d_0, d_avg
    
    def run_comparison_study(self, num_trials: int = 100):
        """
        Run full comparison study
        """
        print("\n" + "="*80)
        print("NAIVE VS STANDARD CP COMPARISON STUDY")
        print("="*80)
        
        # First calibrate tau if not already done
        if self.tau is None:
            self.calibrate_tau(num_calibration_trials=100)
        
        # Use validation environments for testing
        validation_envs = self.cp_config['environments']['full_environments']['validation_envs']
        
        naive_results = []
        standard_results = []
        
        trials_per_env = max(1, num_trials // len(validation_envs))
        
        print(f"\nRunning {num_trials} comparison trials...")
        print(f"Tau = {self.tau:.4f} m")
        
        with tqdm(total=num_trials * 2, desc="Comparison") as pbar:
            for env_dict in validation_envs:
                env_name = env_dict['name']
                env_test_ids = env_dict['test_ids']
                
                parser = self.load_environment(env_name)
                all_env_tests = self.env_config['environments'][env_name]['tests']
                env_tests = [t for t in all_env_tests if t['id'] in env_test_ids]
                
                for trial_idx in range(trials_per_env):
                    test = env_tests[trial_idx % len(env_tests)]
                    seed = trial_idx + 1000  # Different seeds from calibration
                    noise_level = np.random.choice(self.cp_config['noise_model']['noise_levels'])
                    
                    # Run naive planner
                    naive_result = self.run_naive_planner(
                        env_name, test, noise_level, seed
                    )
                    naive_results.append(naive_result)
                    pbar.update(1)
                    
                    # Run Standard CP planner
                    standard_result = self.run_standard_cp_planner(
                        env_name, test, noise_level, seed
                    )
                    standard_results.append(standard_result)
                    pbar.update(1)
        
        # Analyze results
        self.results['naive'] = self.analyze_method_results(naive_results)
        self.results['standard_cp'] = self.analyze_method_results(standard_results)
        self.results['comparison'] = self.compare_methods()
        
        # Save results
        self.save_results()
        self.create_comparison_plots()
        
        return self.results
    
    def analyze_method_results(self, results: List[Dict]) -> Dict:
        """
        Analyze results for a single method
        """
        if not results:
            return {}
        
        planning_successes = [r['planning_success'] for r in results]
        collisions = [r['collision'] for r in results if r['planning_success']]
        d_0_values = [r['d_0'] for r in results if r['d_0'] is not None]
        d_avg_values = [r['d_avg'] for r in results if r['d_avg'] is not None]
        path_lengths = [r['path_length'] for r in results if r['path_length'] > 0]
        
        analysis = {
            'num_trials': len(results),
            'planning_success_rate': np.mean(planning_successes),
            'collision_rate': np.mean(collisions) if collisions else 0.0,
            'd_0': {
                'mean': np.mean(d_0_values) if d_0_values else 0.0,
                'std': np.std(d_0_values) if d_0_values else 0.0,
                'min': np.min(d_0_values) if d_0_values else 0.0,
                'max': np.max(d_0_values) if d_0_values else 0.0
            },
            'd_avg': {
                'mean': np.mean(d_avg_values) if d_avg_values else 0.0,
                'std': np.std(d_avg_values) if d_avg_values else 0.0,
                'min': np.min(d_avg_values) if d_avg_values else 0.0,
                'max': np.max(d_avg_values) if d_avg_values else 0.0
            },
            'path_length': {
                'mean': np.mean(path_lengths) if path_lengths else 0.0,
                'std': np.std(path_lengths) if path_lengths else 0.0
            }
        }
        
        return analysis
    
    def compare_methods(self) -> Dict:
        """
        Generate comparison metrics
        """
        naive = self.results['naive']
        standard = self.results['standard_cp']
        
        comparison = {
            'planning_success_improvement': 
                (standard['planning_success_rate'] - naive['planning_success_rate']) * 100,
            'collision_reduction': 
                (naive['collision_rate'] - standard['collision_rate']) * 100,
            'd_0_improvement': 
                standard['d_0']['mean'] - naive['d_0']['mean'],
            'd_avg_improvement': 
                standard['d_avg']['mean'] - naive['d_avg']['mean'],
            'path_length_increase': 
                (standard['path_length']['mean'] - naive['path_length']['mean']) / 
                naive['path_length']['mean'] * 100 if naive['path_length']['mean'] > 0 else 0
        }
        
        return comparison
    
    def create_comparison_plots(self):
        """
        Create comparison visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        methods = ['Naive', 'Standard CP']
        colors = ['#FF6B6B', '#4ECDC4']
        
        # Plot 1: Success Rate
        ax = axes[0, 0]
        success_rates = [
            self.results['naive']['planning_success_rate'] * 100,
            self.results['standard_cp']['planning_success_rate'] * 100
        ]
        ax.bar(methods, success_rates, color=colors)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Planning Success Rate')
        ax.set_ylim([0, 100])
        
        # Plot 2: Collision Rate
        ax = axes[0, 1]
        collision_rates = [
            self.results['naive']['collision_rate'] * 100,
            self.results['standard_cp']['collision_rate'] * 100
        ]
        ax.bar(methods, collision_rates, color=colors)
        ax.set_ylabel('Collision Rate (%)')
        ax.set_title('Collision Rate (Lower is Better)')
        ax.set_ylim([0, max(collision_rates) * 1.2 if max(collision_rates) > 0 else 1])
        
        # Plot 3: Minimum Clearance (d_0)
        ax = axes[0, 2]
        d_0_means = [
            self.results['naive']['d_0']['mean'],
            self.results['standard_cp']['d_0']['mean']
        ]
        d_0_stds = [
            self.results['naive']['d_0']['std'],
            self.results['standard_cp']['d_0']['std']
        ]
        ax.bar(methods, d_0_means, yerr=d_0_stds, color=colors, capsize=5)
        ax.set_ylabel('Distance (m)')
        ax.set_title('Minimum Clearance (d₀)')
        
        # Plot 4: Average Clearance (d_avg)
        ax = axes[1, 0]
        d_avg_means = [
            self.results['naive']['d_avg']['mean'],
            self.results['standard_cp']['d_avg']['mean']
        ]
        d_avg_stds = [
            self.results['naive']['d_avg']['std'],
            self.results['standard_cp']['d_avg']['std']
        ]
        ax.bar(methods, d_avg_means, yerr=d_avg_stds, color=colors, capsize=5)
        ax.set_ylabel('Distance (m)')
        ax.set_title('Average Clearance (d_avg)')
        
        # Plot 5: Path Length
        ax = axes[1, 1]
        path_means = [
            self.results['naive']['path_length']['mean'],
            self.results['standard_cp']['path_length']['mean']
        ]
        path_stds = [
            self.results['naive']['path_length']['std'],
            self.results['standard_cp']['path_length']['std']
        ]
        ax.bar(methods, path_means, yerr=path_stds, color=colors, capsize=5)
        ax.set_ylabel('Path Length (waypoints)')
        ax.set_title('Path Length')
        
        # Plot 6: Summary Metrics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""IMPROVEMENT METRICS:
        
Collision Reduction: {self.results['comparison']['collision_reduction']:.1f}%
        
Min Clearance Improvement: {self.results['comparison']['d_0_improvement']:.3f} m
        
Avg Clearance Improvement: {self.results['comparison']['d_avg_improvement']:.3f} m
        
Path Length Increase: {self.results['comparison']['path_length_increase']:.1f}%
        
Tau (Safety Margin): {self.tau:.4f} m"""
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Naive vs Standard CP Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f'comparison_plot_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'comparison_plot_{timestamp}.pdf', 
                   format='pdf', bbox_inches='tight')
        
        print(f"\nComparison plot saved to {plot_path}")
        plt.show()
    
    def save_results(self):
        """
        Save results to files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results dictionary
        results = {
            'study': 'Naive vs Standard CP Comparison',
            'timestamp': timestamp,
            'tau': self.tau,
            'results': self.results
        }
        
        # Save as JSON
        json_path = self.results_dir / f'comparison_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"\nResults saved to {json_path}")
        
        # Create comparison table
        df_data = [
            {
                'Method': 'Naive',
                'Success Rate (%)': f"{self.results['naive']['planning_success_rate']*100:.1f}",
                'Collision Rate (%)': f"{self.results['naive']['collision_rate']*100:.1f}",
                'd₀ (m)': f"{self.results['naive']['d_0']['mean']:.3f} ± {self.results['naive']['d_0']['std']:.3f}",
                'd_avg (m)': f"{self.results['naive']['d_avg']['mean']:.3f} ± {self.results['naive']['d_avg']['std']:.3f}",
                'Path Length': f"{self.results['naive']['path_length']['mean']:.1f} ± {self.results['naive']['path_length']['std']:.1f}"
            },
            {
                'Method': 'Standard CP',
                'Success Rate (%)': f"{self.results['standard_cp']['planning_success_rate']*100:.1f}",
                'Collision Rate (%)': f"{self.results['standard_cp']['collision_rate']*100:.1f}",
                'd₀ (m)': f"{self.results['standard_cp']['d_0']['mean']:.3f} ± {self.results['standard_cp']['d_0']['std']:.3f}",
                'd_avg (m)': f"{self.results['standard_cp']['d_avg']['mean']:.3f} ± {self.results['standard_cp']['d_avg']['std']:.3f}",
                'Path Length': f"{self.results['standard_cp']['path_length']['mean']:.1f} ± {self.results['standard_cp']['path_length']['std']:.1f}"
            }
        ]
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = self.results_dir / f'comparison_table_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Comparison table saved to {csv_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        
        print("\n" + "="*80)
        print("IMPROVEMENT METRICS")
        print("="*80)
        print(f"Collision Reduction: {self.results['comparison']['collision_reduction']:.1f}%")
        print(f"Min Clearance Improvement: {self.results['comparison']['d_0_improvement']:.3f} m")
        print(f"Avg Clearance Improvement: {self.results['comparison']['d_avg_improvement']:.3f} m")
        print(f"Path Length Increase: {self.results['comparison']['path_length_increase']:.1f}%")


if __name__ == "__main__":
    # Run the comparison study
    study = NaiveVsStandardCPComparison()
    study.run_comparison_study(num_trials=100)