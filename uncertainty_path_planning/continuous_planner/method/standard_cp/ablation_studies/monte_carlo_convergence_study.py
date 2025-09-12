#!/usr/bin/env python3
"""
Monte Carlo Convergence Study for Standard CP - ICRA 2025
Analyzes how tau converges as sample size increases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Import components from common area
from noise_model import NoiseModel
from nonconformity_scorer import NonconformityScorer
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid


class MonteCarloConvergenceStudy:
    """
    Study how tau (safety margin) converges with increasing sample sizes
    """
    
    def __init__(self):
        # Sample sizes to test (reduced for faster results)
        self.sample_sizes = [10, 50, 100, 200]  # Reduced range for quick analysis
        
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
        
        # Results storage
        self.results = {}
        self.convergence_data = []
        
        # Create results directory
        self.results_dir = Path('results/ablation_studies/monte_carlo_convergence')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for environments
        self._env_cache = {}
    
    def load_environment(self, env_name: str) -> MRPBMapParser:
        """Load and cache environment"""
        if env_name not in self._env_cache:
            self._env_cache[env_name] = MRPBMapParser(
                map_name=env_name,
                mrpb_path='../../../mrpb_dataset'
            )
        return self._env_cache[env_name]
    
    def run_trials_for_sample_size(self, sample_size: int) -> Dict:
        """
        Run trials with a specific sample size
        """
        print(f"\n{'='*60}")
        print(f"Running Monte Carlo with {sample_size} samples")
        print(f"{'='*60}")
        
        # Use calibration environments
        calibration_envs = self.cp_config['environments']['full_environments']['calibration_envs']
        
        all_scores = []
        trial_details = []
        planning_failures = 0
        
        # Calculate trials per environment
        trials_per_env = max(1, sample_size // len(calibration_envs))
        remaining_trials = sample_size - (trials_per_env * len(calibration_envs))
        
        with tqdm(total=sample_size, desc=f"N={sample_size}") as pbar:
            for env_idx, env_dict in enumerate(calibration_envs):
                env_name = env_dict['name']
                env_test_ids = env_dict['test_ids']
                
                # Add extra trials to first environment if needed
                env_trials = trials_per_env + (remaining_trials if env_idx == 0 else 0)
                
                # Load environment
                parser = self.load_environment(env_name)
                
                # Get test configurations
                all_env_tests = self.env_config['environments'][env_name]['tests']
                # Filter to only use specified test_ids
                env_tests = [t for t in all_env_tests if t['id'] in env_test_ids]
                
                for trial_idx in range(env_trials):
                    # Select test configuration
                    test = env_tests[trial_idx % len(env_tests)]
                    start = test['start']
                    goal = test['goal']
                    
                    # Set seed for reproducibility
                    seed = env_idx * 1000 + trial_idx
                    np.random.seed(seed)
                    
                    # Get true grid
                    true_grid = parser.occupancy_grid.copy()
                    
                    # Sample noise level
                    noise_level = np.random.choice(self.cp_config['noise_model']['noise_levels'])
                    
                    # Add noise to create perceived grid
                    perceived_grid = self.noise_model.add_realistic_noise(
                        true_grid, noise_level, seed=seed
                    )
                    
                    # Plan path on perceived grid
                    planner = RRTStarGrid(
                        start=start,
                        goal=goal,
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
                            # Compute nonconformity score
                            score = self.nonconformity_scorer.compute_nonconformity_score(
                                true_grid, perceived_grid, path, parser
                            )
                            
                            all_scores.append(score)
                            trial_details.append({
                                'env': env_name,
                                'trial': trial_idx,
                                'noise_level': noise_level,
                                'score': score,
                                'path_length': len(path)
                            })
                        else:
                            planning_failures += 1
                            # Add max score for planning failure
                            all_scores.append(0.3)  # Max safety margin
                    
                    except Exception as e:
                        print(f"  Trial {trial_idx} failed: {e}")
                        planning_failures += 1
                        all_scores.append(0.3)
                    
                    pbar.update(1)
        
        # Calculate tau (90th percentile)
        tau = self.calculate_tau(all_scores)
        
        # Calculate confidence interval using bootstrap
        ci_lower, ci_upper = self.bootstrap_confidence_interval(all_scores)
        
        # Statistics
        stats = self.calculate_statistics(all_scores)
        stats['planning_failures'] = planning_failures
        
        result = {
            'sample_size': sample_size,
            'tau': tau,
            'confidence_interval': [ci_lower, ci_upper],
            'statistics': stats,
            'num_scores': len(all_scores),
            'scores_sample': all_scores[:10]  # First 10 for inspection
        }
        
        print(f"\nResults for N={sample_size}:")
        print(f"  Tau: {tau:.4f} m")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Mean score: {stats['mean']:.4f}")
        print(f"  Std dev: {stats['std']:.4f}")
        print(f"  Planning failures: {planning_failures}/{sample_size}")
        
        return result
    
    def calculate_tau(self, scores: List[float], confidence_level: float = 0.9) -> float:
        """Calculate tau (quantile) from scores"""
        if not scores:
            return 0.0
        
        sorted_scores = sorted(scores)
        quantile_idx = int(np.ceil((len(sorted_scores) + 1) * confidence_level)) - 1
        quantile_idx = min(quantile_idx, len(sorted_scores) - 1)
        
        return sorted_scores[quantile_idx]
    
    def bootstrap_confidence_interval(self, scores: List[float], 
                                     n_bootstrap: int = 1000,
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate confidence interval for tau using bootstrap
        """
        if not scores:
            return 0.0, 0.0
        
        bootstrap_taus = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_taus.append(self.calculate_tau(sample.tolist()))
        
        # Calculate percentiles
        lower = np.percentile(bootstrap_taus, alpha/2 * 100)
        upper = np.percentile(bootstrap_taus, (1 - alpha/2) * 100)
        
        return float(lower), float(upper)
    
    def calculate_statistics(self, values: List[float]) -> Dict:
        """Calculate comprehensive statistics"""
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'percentiles': {},
                'zero_scores': 0
            }
        
        arr = np.array(values)
        return {
            'count': len(values),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'percentiles': {
                '25%': float(np.percentile(arr, 25)),
                '50%': float(np.percentile(arr, 50)),
                '75%': float(np.percentile(arr, 75)),
                '90%': float(np.percentile(arr, 90)),
                '95%': float(np.percentile(arr, 95)),
                '99%': float(np.percentile(arr, 99))
            },
            'zero_scores': float(np.sum(arr == 0))
        }
    
    def analyze_convergence(self) -> Dict:
        """
        Analyze convergence behavior
        """
        if not self.convergence_data:
            return {}
        
        # Extract tau values and confidence intervals
        sample_sizes = [d['sample_size'] for d in self.convergence_data]
        taus = [d['tau'] for d in self.convergence_data]
        ci_lowers = [d['confidence_interval'][0] for d in self.convergence_data]
        ci_uppers = [d['confidence_interval'][1] for d in self.convergence_data]
        
        # Check for convergence (when tau stabilizes)
        converged = False
        convergence_sample_size = None
        convergence_threshold = 0.001  # 1mm difference
        
        for i in range(1, len(taus)):
            if abs(taus[i] - taus[i-1]) < convergence_threshold:
                converged = True
                convergence_sample_size = sample_sizes[i]
                break
        
        # Calculate convergence rate
        convergence_rates = []
        for i in range(1, len(taus)):
            rate = abs(taus[i] - taus[i-1]) / (sample_sizes[i] - sample_sizes[i-1])
            convergence_rates.append(rate)
        
        analysis = {
            'converged': converged,
            'convergence_sample_size': convergence_sample_size,
            'final_tau': taus[-1],
            'tau_range': [min(taus), max(taus)],
            'ci_width_trend': [ci_uppers[i] - ci_lowers[i] for i in range(len(ci_lowers))],
            'convergence_rates': convergence_rates,
            'recommendation': self._get_recommendation(converged, convergence_sample_size)
        }
        
        return analysis
    
    def _get_recommendation(self, converged: bool, convergence_size: int) -> str:
        """Generate recommendation based on convergence analysis"""
        if not converged:
            return "Tau has not converged. Consider using more than 2000 samples."
        elif convergence_size <= 200:
            return f"Tau converges quickly at {convergence_size} samples. {convergence_size}-500 samples recommended for efficiency."
        elif convergence_size <= 500:
            return f"Tau converges at {convergence_size} samples. 500-1000 samples recommended for robustness."
        else:
            return f"Tau requires {convergence_size} samples to converge. Use at least {convergence_size} samples."
    
    def create_convergence_plot(self):
        """
        Create publication-quality convergence plot
        """
        if not self.convergence_data:
            print("No data to plot")
            return
        
        # Extract data
        sample_sizes = [d['sample_size'] for d in self.convergence_data]
        taus = [d['tau'] for d in self.convergence_data]
        ci_lowers = [d['confidence_interval'][0] for d in self.convergence_data]
        ci_uppers = [d['confidence_interval'][1] for d in self.convergence_data]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Tau convergence with confidence intervals
        ax1.plot(sample_sizes, taus, 'b-', linewidth=2, label='τ (90th percentile)', marker='o')
        ax1.fill_between(sample_sizes, ci_lowers, ci_uppers, 
                         alpha=0.3, color='blue', label='95% CI')
        
        # Add horizontal line for final tau
        ax1.axhline(y=taus[-1], color='r', linestyle='--', alpha=0.5, 
                   label=f'Converged τ = {taus[-1]:.4f}m')
        
        ax1.set_xlabel('Sample Size', fontsize=12)
        ax1.set_ylabel('τ (meters)', fontsize=12)
        ax1.set_title('Monte Carlo Convergence of Safety Margin τ', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Plot 2: Confidence interval width
        ci_widths = [ci_uppers[i] - ci_lowers[i] for i in range(len(ci_lowers))]
        ax2.plot(sample_sizes, ci_widths, 'g-', linewidth=2, marker='s')
        
        ax2.set_xlabel('Sample Size', fontsize=12)
        ax2.set_ylabel('CI Width (meters)', fontsize=12)
        ax2.set_title('Confidence Interval Width vs Sample Size', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f'convergence_plot_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'convergence_plot_{timestamp}.pdf', 
                   format='pdf', bbox_inches='tight')
        
        print(f"\nConvergence plot saved to {plot_path}")
        plt.show()
    
    def save_results(self):
        """
        Save all results to files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results dictionary
        results = {
            'study': 'Monte Carlo Convergence Analysis',
            'timestamp': timestamp,
            'sample_sizes': self.sample_sizes,
            'convergence_data': self.convergence_data,
            'analysis': self.analyze_convergence()
        }
        
        # Save as JSON
        json_path = self.results_dir / f'convergence_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {json_path}")
        
        # Create summary DataFrame
        df_data = []
        for d in self.convergence_data:
            df_data.append({
                'Sample Size': d['sample_size'],
                'Tau (m)': f"{d['tau']:.4f}",
                'CI Lower': f"{d['confidence_interval'][0]:.4f}",
                'CI Upper': f"{d['confidence_interval'][1]:.4f}",
                'CI Width': f"{d['confidence_interval'][1] - d['confidence_interval'][0]:.4f}",
                'Mean Score': f"{d['statistics']['mean']:.4f}",
                'Std Dev': f"{d['statistics']['std']:.4f}",
                'Planning Failures': d['statistics'].get('planning_failures', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        csv_path = self.results_dir / f'convergence_summary_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to {csv_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("CONVERGENCE SUMMARY TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        # Print analysis
        analysis = self.analyze_convergence()
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS")
        print("="*80)
        print(f"Converged: {analysis['converged']}")
        if analysis['converged']:
            print(f"Convergence at: {analysis['convergence_sample_size']} samples")
        print(f"Final tau: {analysis['final_tau']:.4f} m")
        print(f"Tau range: [{analysis['tau_range'][0]:.4f}, {analysis['tau_range'][1]:.4f}] m")
        print(f"\nRecommendation: {analysis['recommendation']}")
    
    def run_full_study(self):
        """
        Run the complete Monte Carlo convergence study
        """
        print("\n" + "="*80)
        print("MONTE CARLO CONVERGENCE STUDY FOR STANDARD CP")
        print("Testing sample sizes:", self.sample_sizes)
        print("="*80)
        
        # Run for each sample size
        for sample_size in self.sample_sizes:
            result = self.run_trials_for_sample_size(sample_size)
            self.convergence_data.append(result)
            
            # Save intermediate results
            self.save_results()
        
        # Create visualization
        self.create_convergence_plot()
        
        # Final save
        self.save_results()
        
        print("\n" + "="*80)
        print("STUDY COMPLETE")
        print("="*80)


if __name__ == "__main__":
    # Run the Monte Carlo convergence study
    study = MonteCarloConvergenceStudy()
    study.run_full_study()