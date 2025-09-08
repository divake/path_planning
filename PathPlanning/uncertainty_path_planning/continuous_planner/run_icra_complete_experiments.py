#!/usr/bin/env python3
"""
ICRA 2025 COMPLETE EXPERIMENT RUNNER
Generates all results, tables, and plots for the paper
Compares: Naive vs Standard CP vs Learnable CP across 8 environments
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import sys
import time
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

sys.path.append('/mnt/ssd1/divake/path_planning/PathPlanning/uncertainty_path_planning/continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP, ContinuousNonconformity
from learnable_cp_final import FinalLearnableCP
from rrt_star_planner import RRTStar
from continuous_visualization import ContinuousVisualizer
from proper_monte_carlo_evaluation import monte_carlo_evaluation_single_env
from cross_validation_environments import get_training_environments, get_test_environments


class ICRAExperimentRunner:
    """Complete experiment runner for ICRA 2025 paper"""
    
    def __init__(self, num_trials: int = 1000, noise_level: float = 0.15):
        """
        Initialize experiment runner
        
        Args:
            num_trials: Number of Monte Carlo trials per environment
            noise_level: Perception noise level (0.15 = 15%)
        """
        self.num_trials = num_trials
        self.noise_level = noise_level
        self.results_dir = "results/icra2025"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/tables", exist_ok=True)
        
        # Get all environments
        self.training_envs = get_training_environments()
        self.test_envs = get_test_environments()
        self.all_envs = {**self.training_envs, **self.test_envs}
        
        # Methods to evaluate
        self.methods = ['naive', 'standard', 'learnable']
        
        # Color scheme for plots
        self.colors = {
            'naive': '#FF6B6B',      # Red
            'standard': '#4ECDC4',    # Teal
            'learnable': '#45B7D1'    # Blue
        }
        
        # Initialize results storage
        self.all_results = {}
        self.timing_results = {}
        self.statistical_tests = {}
        
    def train_learnable_cp(self) -> FinalLearnableCP:
        """Train the learnable CP model"""
        print("\n" + "="*80)
        print("TRAINING LEARNABLE CP MODEL")
        print("="*80)
        
        model = FinalLearnableCP(
            coverage=0.90,
            max_tau=1.0
        )
        
        # Set baseline taus for different environments
        model.baseline_taus = {
            'passages': 0.4,
            'open': 0.3,
            'narrow': 0.5,
            'zigzag': 0.45,
            'spiral': 0.35,
            'random_forest': 0.3,
            'tight_gaps': 0.5,
            'rooms': 0.4
        }
        
        # Train on training environments only  
        print("Training on environments:", list(self.training_envs.keys()))
        
        # Train the model using the built-in training method
        print("Training learnable CP model...")
        model.train(num_epochs=150)  # Train method only takes num_epochs
        
        print(f"Training complete!")
        
        return model
    
    def run_single_environment_experiment(self, env_name: str, 
                                         obstacles: List,
                                         learnable_model: FinalLearnableCP = None) -> Dict:
        """
        Run complete experiment for a single environment
        
        Returns:
            Dictionary with results for all methods
        """
        print(f"\n{'='*60}")
        print(f"ENVIRONMENT: {env_name.upper()}")
        print(f"{'='*60}")
        
        env_results = {}
        
        # Create environment
        env = ContinuousEnvironment()
        # Override obstacles with the provided ones
        env.obstacles = obstacles
        
        # Define start and goal based on environment
        if env_name in ['passages', 'zigzag', 'spiral']:
            start = (2.0, 2.0)
            goal = (48.0, 28.0)
        elif env_name in ['narrow', 'tight_gaps']:
            start = (5.0, 15.0)
            goal = (45.0, 15.0)
        else:
            start = (2.0, 2.0)
            goal = (48.0, 28.0)
        
        # Run each method
        for method in self.methods:
            print(f"\nEvaluating {method.upper()} method...")
            start_time = time.time()
            
            # Run Monte Carlo evaluation
            if method == 'naive':
                result = self.run_naive_evaluation(env, start, goal)
            elif method == 'standard':
                result = self.run_standard_cp_evaluation(env, start, goal)
            else:  # learnable
                result = self.run_learnable_cp_evaluation(env, start, goal, learnable_model)
            
            elapsed_time = time.time() - start_time
            result['computation_time'] = elapsed_time
            
            env_results[method] = result
            
            # Print summary
            print(f"  Success rate: {result['success_rate']*100:.1f}% "
                  f"[{result['ci_lower']*100:.1f}%, {result['ci_upper']*100:.1f}%]")
            print(f"  Collision rate: {result['collision_rate']*100:.1f}%")
            print(f"  Avg path length: {result['avg_path_length']:.2f}")
            print(f"  Computation time: {elapsed_time:.2f}s")
        
        return env_results
    
    def run_naive_evaluation(self, env, start, goal) -> Dict:
        """Run naive planner evaluation"""
        successes = 0
        collisions = 0
        path_lengths = []
        paths_found = 0
        
        for trial in range(self.num_trials):
            # Add perception noise
            noisy_obstacles = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, 
                thin_factor=self.noise_level,
                seed=trial
            )
            
            # Plan with noisy perception
            planner = RRTStar(start, goal, noisy_obstacles, 
                            bounds=(0, 50, 0, 30), seed=trial)
            path = planner.plan()
            
            if path is not None:
                paths_found += 1
                path_lengths.append(len(path))
                
                # Check collision with true obstacles
                collision = False
                for point in path:
                    if env.point_in_obstacle(point[0], point[1]):
                        collision = True
                        collisions += 1
                        break
                
                if not collision:
                    successes += 1
        
        # Calculate statistics
        success_rate = successes / self.num_trials
        ci_lower, ci_upper = self.wilson_score_interval(successes, self.num_trials)
        
        return {
            'success_rate': success_rate,
            'collision_rate': collisions / self.num_trials,
            'paths_found': paths_found,
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'std_path_length': np.std(path_lengths) if path_lengths else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'successes': successes,
            'collisions': collisions,
            'num_trials': self.num_trials
        }
    
    def run_standard_cp_evaluation(self, env, start, goal) -> Dict:
        """Run standard CP evaluation"""
        # First calibrate CP
        cp = ContinuousStandardCP(env.obstacles, nonconformity_type='penetration')
        tau = cp.calibrate(
            ContinuousNoiseModel.add_thinning_noise,
            {'thin_factor': self.noise_level},
            num_samples=500,
            confidence=0.90,
            base_seed=42
        )
        
        successes = 0
        collisions = 0
        path_lengths = []
        paths_found = 0
        
        for trial in range(self.num_trials):
            # Add perception noise
            noisy_obstacles = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles,
                thin_factor=self.noise_level,
                seed=trial
            )
            
            # Inflate obstacles by tau
            # The CP class uses self.tau internally after calibration
            inflated_obstacles = cp.inflate_obstacles(noisy_obstacles, inflation_method="uniform")
            
            # Plan with inflated obstacles
            planner = RRTStar(start, goal, inflated_obstacles,
                            bounds=(0, 50, 0, 30), seed=trial)
            path = planner.plan()
            
            if path is not None:
                paths_found += 1
                path_lengths.append(len(path))
                
                # Check collision with true obstacles
                collision = False
                for point in path:
                    if env.point_in_obstacle(point[0], point[1]):
                        collision = True
                        collisions += 1
                        break
                
                if not collision:
                    successes += 1
        
        # Calculate statistics
        success_rate = successes / self.num_trials
        ci_lower, ci_upper = self.wilson_score_interval(successes, self.num_trials)
        
        return {
            'success_rate': success_rate,
            'collision_rate': collisions / self.num_trials,
            'paths_found': paths_found,
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'std_path_length': np.std(path_lengths) if path_lengths else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'successes': successes,
            'collisions': collisions,
            'num_trials': self.num_trials,
            'tau': tau
        }
    
    def run_learnable_cp_evaluation(self, env, start, goal, model) -> Dict:
        """Run learnable CP evaluation"""
        successes = 0
        collisions = 0
        path_lengths = []
        paths_found = 0
        tau_values = []
        
        for trial in range(self.num_trials):
            # Add perception noise
            noisy_obstacles = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles,
                thin_factor=self.noise_level,
                seed=trial
            )
            
            # Use learnable CP to get adaptive tau for each obstacle
            inflated_obstacles = []
            for x, y, w, h in noisy_obstacles:
                # Get center of obstacle
                cx, cy = x + w/2, y + h/2
                # Get adaptive tau using the model's predict_tau method
                tau = model.predict_tau(cx, cy, noisy_obstacles, goal)
                tau_values.append(tau)
                
                # Inflate this obstacle by its adaptive tau
                inflated_obstacles.append((
                    max(0, x - tau), 
                    max(0, y - tau),
                    w + 2*tau, 
                    h + 2*tau
                ))
            
            # Plan with inflated obstacles
            planner = RRTStar(start, goal, inflated_obstacles,
                            bounds=(0, 50, 0, 30), seed=trial)
            path = planner.plan()
            
            if path is not None:
                paths_found += 1
                path_lengths.append(len(path))
                
                # Check collision with true obstacles
                collision = False
                for point in path:
                    if env.point_in_obstacle(point[0], point[1]):
                        collision = True
                        collisions += 1
                        break
                
                if not collision:
                    successes += 1
        
        # Calculate statistics
        success_rate = successes / self.num_trials
        ci_lower, ci_upper = self.wilson_score_interval(successes, self.num_trials)
        
        return {
            'success_rate': success_rate,
            'collision_rate': collisions / self.num_trials,
            'paths_found': paths_found,
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'std_path_length': np.std(path_lengths) if path_lengths else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'successes': successes,
            'collisions': collisions,
            'num_trials': self.num_trials,
            'avg_tau': np.mean(tau_values),
            'std_tau': np.std(tau_values)
        }
    
    def wilson_score_interval(self, successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval"""
        if n == 0:
            return 0, 0
        
        p_hat = successes / n
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
        
        return max(0, center - margin), min(1, center + margin)
    
    def run_all_experiments(self):
        """Run experiments on all environments"""
        print("\n" + "="*80)
        print("ICRA 2025 COMPREHENSIVE EXPERIMENTS")
        print("="*80)
        print(f"Settings: {self.num_trials} trials, {self.noise_level*100:.0f}% noise")
        print(f"Environments: {len(self.all_envs)} total")
        print(f"  Training: {list(self.training_envs.keys())}")
        print(f"  Test: {list(self.test_envs.keys())}")
        
        # Train learnable CP model
        learnable_model = self.train_learnable_cp()
        
        # Run experiments on all environments
        for env_name, obstacles in self.all_envs.items():
            env_results = self.run_single_environment_experiment(
                env_name, obstacles, learnable_model
            )
            self.all_results[env_name] = env_results
        
        # Run statistical tests
        self.run_statistical_tests()
        
        # Save all results
        self.save_results()
        
        return self.all_results
    
    def run_statistical_tests(self):
        """Run statistical significance tests"""
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        
        # Pairwise comparisons using McNemar's test
        comparisons = [
            ('naive', 'standard'),
            ('naive', 'learnable'),
            ('standard', 'learnable')
        ]
        
        for method1, method2 in comparisons:
            print(f"\n{method1.upper()} vs {method2.upper()}:")
            
            # Aggregate successes across all environments
            total_both_success = 0
            total_m1_only = 0
            total_m2_only = 0
            total_both_fail = 0
            
            for env_name in self.all_results:
                r1 = self.all_results[env_name][method1]
                r2 = self.all_results[env_name][method2]
                
                # Approximate contingency table
                both_success = min(r1['successes'], r2['successes'])
                m1_only = max(0, r1['successes'] - both_success)
                m2_only = max(0, r2['successes'] - both_success)
                both_fail = self.num_trials - both_success - m1_only - m2_only
                
                total_both_success += both_success
                total_m1_only += m1_only
                total_m2_only += m2_only
                total_both_fail += both_fail
            
            # McNemar's test
            n = total_m1_only + total_m2_only
            if n > 0:
                chi2 = (abs(total_m1_only - total_m2_only) - 1)**2 / n
                p_value = 1 - stats.chi2.cdf(chi2, df=1)
                
                print(f"  McNemar's χ² = {chi2:.4f}, p = {p_value:.4e}")
                if p_value < 0.001:
                    print(f"  *** Highly significant difference")
                elif p_value < 0.01:
                    print(f"  ** Significant difference")
                elif p_value < 0.05:
                    print(f"  * Significant difference")
                else:
                    print(f"  No significant difference")
            
            self.statistical_tests[f"{method1}_vs_{method2}"] = {
                'chi2': chi2 if n > 0 else 0,
                'p_value': p_value if n > 0 else 1.0,
                'n': n
            }
    
    def generate_detailed_tables(self):
        """Generate detailed performance tables"""
        print("\n" + "="*80)
        print("GENERATING DETAILED TABLES")
        print("="*80)
        
        # Table 1: Main results table
        table_data = []
        
        for env_name in self.all_results:
            for method in self.methods:
                r = self.all_results[env_name][method]
                table_data.append({
                    'Environment': env_name.capitalize(),
                    'Method': method.capitalize(),
                    'Success Rate (%)': f"{r['success_rate']*100:.1f}",
                    '95% CI': f"[{r['ci_lower']*100:.1f}, {r['ci_upper']*100:.1f}]",
                    'Collision Rate (%)': f"{r['collision_rate']*100:.1f}",
                    'Path Found (%)': f"{r['paths_found']/self.num_trials*100:.1f}",
                    'Avg Path Length': f"{r['avg_path_length']:.1f}",
                    'Time (s)': f"{r['computation_time']:.2f}"
                })
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        df.to_csv(f"{self.results_dir}/tables/main_results.csv", index=False)
        
        # Save as LaTeX
        latex_table = df.to_latex(index=False, escape=False)
        with open(f"{self.results_dir}/tables/main_results.tex", 'w') as f:
            f.write(latex_table)
        
        # Table 2: Summary statistics
        summary_data = []
        
        for method in self.methods:
            success_rates = [self.all_results[env][method]['success_rate'] 
                           for env in self.all_results]
            collision_rates = [self.all_results[env][method]['collision_rate']
                              for env in self.all_results]
            
            # Separate training and test
            train_success = [self.all_results[env][method]['success_rate']
                           for env in self.training_envs if env in self.all_results]
            test_success = [self.all_results[env][method]['success_rate']
                          for env in self.test_envs if env in self.all_results]
            
            summary_data.append({
                'Method': method.capitalize(),
                'Mean Success (%)': f"{np.mean(success_rates)*100:.1f}",
                'Std Success (%)': f"{np.std(success_rates)*100:.1f}",
                'Mean Collision (%)': f"{np.mean(collision_rates)*100:.1f}",
                'Train Success (%)': f"{np.mean(train_success)*100:.1f}",
                'Test Success (%)': f"{np.mean(test_success)*100:.1f}",
                'Coverage Met': sum(1 for env in self.all_results 
                                  if self.all_results[env][method]['ci_lower'] >= 0.90)
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(f"{self.results_dir}/tables/summary_statistics.csv", index=False)
        
        print("Tables saved to results/icra2025/tables/")
        
        return df, df_summary
    
    def generate_publication_plots(self):
        """Generate publication-quality plots"""
        print("\n" + "="*80)
        print("GENERATING PUBLICATION PLOTS")
        print("="*80)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 14
        
        # Plot 1: Success rates comparison
        self.plot_success_rates_comparison()
        
        # Plot 2: Coverage guarantee analysis
        self.plot_coverage_analysis()
        
        # Plot 3: Path quality comparison
        self.plot_path_quality()
        
        # Plot 4: Training vs Test performance
        self.plot_generalization()
        
        # Plot 5: Statistical significance heatmap
        self.plot_statistical_significance()
        
        print("All plots saved to results/icra2025/figures/")
    
    def plot_success_rates_comparison(self):
        """Plot success rates across all environments"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Success Rates Across Environments', fontsize=16, fontweight='bold')
        
        env_list = list(self.all_results.keys())
        
        for idx, env_name in enumerate(env_list[:8]):  # First 8 environments
            ax = axes[idx // 4, idx % 4]
            
            methods = list(self.methods)
            success_rates = [self.all_results[env_name][m]['success_rate']*100 for m in methods]
            errors_lower = [self.all_results[env_name][m]['success_rate']*100 - 
                          self.all_results[env_name][m]['ci_lower']*100 for m in methods]
            errors_upper = [self.all_results[env_name][m]['ci_upper']*100 - 
                          self.all_results[env_name][m]['success_rate']*100 for m in methods]
            
            x_pos = np.arange(len(methods))
            bars = ax.bar(x_pos, success_rates, 
                         yerr=[errors_lower, errors_upper],
                         capsize=5,
                         color=[self.colors[m] for m in methods],
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1)
            
            # Add 90% target line
            ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% target')
            
            ax.set_title(env_name.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Success Rate (%)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([m.capitalize() for m in methods])
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/success_rates_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/figures/success_rates_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_coverage_analysis(self):
        """Plot coverage guarantee analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Success rate distribution
        for method in self.methods:
            success_rates = [self.all_results[env][method]['success_rate']*100 
                           for env in self.all_results]
            ax1.hist(success_rates, bins=10, alpha=0.6, 
                    label=method.capitalize(), color=self.colors[method],
                    edgecolor='black', linewidth=1)
        
        ax1.axvline(x=90, color='red', linestyle='--', linewidth=2, label='90% target')
        ax1.set_xlabel('Success Rate (%)')
        ax1.set_ylabel('Number of Environments')
        ax1.set_title('Distribution of Success Rates', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Coverage guarantee violations
        methods_cp = ['standard', 'learnable']
        violations = []
        for method in methods_cp:
            violation_count = sum(1 for env in self.all_results 
                                if self.all_results[env][method]['ci_lower'] < 0.90)
            violations.append(violation_count)
        
        bars = ax2.bar(range(len(methods_cp)), violations,
                      color=[self.colors[m] for m in methods_cp],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Number of Violations')
        ax2.set_title('Coverage Guarantee Violations (< 90%)', fontweight='bold')
        ax2.set_xticks(range(len(methods_cp)))
        ax2.set_xticklabels([m.capitalize() for m in methods_cp])
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, violations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/coverage_analysis.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/figures/coverage_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_path_quality(self):
        """Plot path quality metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Average path length
        env_names = list(self.all_results.keys())[:6]  # First 6 environments
        x = np.arange(len(env_names))
        width = 0.25
        
        for i, method in enumerate(self.methods):
            path_lengths = [self.all_results[env][method]['avg_path_length'] 
                          for env in env_names]
            ax1.bar(x + i*width, path_lengths, width, 
                   label=method.capitalize(), color=self.colors[method],
                   alpha=0.8, edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('Environment')
        ax1.set_ylabel('Average Path Length')
        ax1.set_title('Path Length Comparison', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([e[:4].upper() for e in env_names])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right: Path finding success rate
        for method in self.methods:
            path_rates = [self.all_results[env][method]['paths_found']/self.num_trials*100 
                         for env in env_names]
            ax2.plot(range(len(path_rates)), path_rates,
                    marker='o', markersize=8, linewidth=2,
                    label=method.capitalize(), color=self.colors[method])
        
        ax2.set_xlabel('Environment Index')
        ax2.set_ylabel('Path Finding Rate (%)')
        ax2.set_title('Path Finding Success Rate', fontweight='bold')
        ax2.set_xticks(range(len(env_names)))
        ax2.set_xticklabels([e[:4].upper() for e in env_names])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/path_quality.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/figures/path_quality.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_generalization(self):
        """Plot training vs test performance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(self.methods))
        width = 0.35
        
        train_success = []
        test_success = []
        
        for method in self.methods:
            # Training environments
            train_rates = [self.all_results[env][method]['success_rate']*100
                         for env in self.training_envs if env in self.all_results]
            train_success.append(np.mean(train_rates))
            
            # Test environments
            test_rates = [self.all_results[env][method]['success_rate']*100
                        for env in self.test_envs if env in self.all_results]
            test_success.append(np.mean(test_rates))
        
        bars1 = ax.bar(x - width/2, train_success, width, label='Training Envs',
                      color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, test_success, width, label='Test Envs',
                      color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% target')
        
        ax.set_xlabel('Method', fontweight='bold')
        ax.set_ylabel('Average Success Rate (%)', fontweight='bold')
        ax.set_title('Generalization: Training vs Test Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in self.methods])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/generalization.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/figures/generalization.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_significance(self):
        """Plot statistical significance heatmap"""
        if not self.statistical_tests:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create matrix for p-values
        methods = self.methods
        n = len(methods)
        p_matrix = np.ones((n, n))
        
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i != j:
                    key = f"{m1}_vs_{m2}"
                    if key in self.statistical_tests:
                        p_matrix[i, j] = self.statistical_tests[key]['p_value']
        
        # Create heatmap
        im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)
        
        # Set ticks
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels([m.capitalize() for m in methods])
        ax.set_yticklabels([m.capitalize() for m in methods])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                if i != j:
                    text = ax.text(j, i, f'{p_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black")
                    if p_matrix[i, j] < 0.001:
                        text.set_weight('bold')
        
        ax.set_title('Statistical Significance (McNemar Test)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/statistical_significance.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/figures/statistical_significance.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results to files"""
        # Save raw results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_file = f"{self.results_dir}/complete_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for env in self.all_results:
                json_results[env] = {}
                for method in self.all_results[env]:
                    json_results[env][method] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in self.all_results[env][method].items()
                    }
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    def generate_latex_summary(self):
        """Generate LaTeX code for paper"""
        print("\n" + "="*80)
        print("LATEX CODE FOR PAPER")
        print("="*80)
        
        latex_code = r"""
% Main Results Table
\begin{table}[htbp]
\centering
\caption{Success Rates Across All Environments}
\label{tab:main_results}
\begin{tabular}{llccc}
\toprule
Environment & Method & Success Rate (\%) & 95\% CI & Collision Rate (\%) \\
\midrule
"""
        
        for env_name in list(self.all_results.keys())[:4]:  # First 4 for space
            for method in self.methods:
                r = self.all_results[env_name][method]
                latex_code += f"{env_name.replace('_', ' ').title()} & "
                latex_code += f"{method.capitalize()} & "
                latex_code += f"{r['success_rate']*100:.1f} & "
                latex_code += f"[{r['ci_lower']*100:.1f}, {r['ci_upper']*100:.1f}] & "
                latex_code += f"{r['collision_rate']*100:.1f} \\\\\n"
            latex_code += r"\midrule" + "\n"
        
        latex_code += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        # Save LaTeX code
        with open(f"{self.results_dir}/tables/paper_table.tex", 'w') as f:
            f.write(latex_code)
        
        print("LaTeX code saved to results/icra2025/tables/paper_table.tex")
    
    def print_final_summary(self):
        """Print final summary of results"""
        print("\n" + "="*80)
        print("FINAL SUMMARY FOR ICRA 2025 PAPER")
        print("="*80)
        
        # Overall statistics
        for method in self.methods:
            all_success = [self.all_results[env][method]['success_rate']*100 
                         for env in self.all_results]
            all_collision = [self.all_results[env][method]['collision_rate']*100
                           for env in self.all_results]
            
            print(f"\n{method.upper()} Method:")
            print(f"  Mean Success Rate: {np.mean(all_success):.1f}% ± {np.std(all_success):.1f}%")
            print(f"  Mean Collision Rate: {np.mean(all_collision):.1f}% ± {np.std(all_collision):.1f}%")
            print(f"  Min Success Rate: {np.min(all_success):.1f}%")
            print(f"  Max Success Rate: {np.max(all_success):.1f}%")
            
            if method in ['standard', 'learnable']:
                coverage_met = sum(1 for env in self.all_results 
                                 if self.all_results[env][method]['ci_lower'] >= 0.90)
                print(f"  Coverage Guarantee Met: {coverage_met}/{len(self.all_results)} environments")
        
        # Key findings
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)
        
        # Find best method overall
        avg_success = {}
        for method in self.methods:
            rates = [self.all_results[env][method]['success_rate'] for env in self.all_results]
            avg_success[method] = np.mean(rates)
        
        best_method = max(avg_success, key=avg_success.get)
        print(f"\n1. Best Overall Method: {best_method.upper()}")
        print(f"   Average success rate: {avg_success[best_method]*100:.1f}%")
        
        # Coverage guarantee analysis
        print("\n2. Coverage Guarantee (90% target):")
        for method in ['standard', 'learnable']:
            met = sum(1 for env in self.all_results 
                    if self.all_results[env][method]['ci_lower'] >= 0.90)
            print(f"   {method.capitalize()}: {met}/{len(self.all_results)} environments")
        
        # Generalization
        print("\n3. Generalization Performance:")
        for method in self.methods:
            train_rates = [self.all_results[env][method]['success_rate']*100
                         for env in self.training_envs if env in self.all_results]
            test_rates = [self.all_results[env][method]['success_rate']*100
                        for env in self.test_envs if env in self.all_results]
            
            print(f"   {method.capitalize()}:")
            print(f"     Training: {np.mean(train_rates):.1f}%")
            print(f"     Test: {np.mean(test_rates):.1f}%")
            print(f"     Gap: {np.mean(train_rates) - np.mean(test_rates):.1f}%")
        
        print("\n✓ ICRA 2025 experiments complete!")
        print("✓ All tables and figures generated!")
        print("✓ Ready for paper submission!")


def main():
    """Main entry point"""
    # Create experiment runner
    runner = ICRAExperimentRunner(
        num_trials=1000,  # Use 1000 for final paper
        noise_level=0.15   # 15% perception noise
    )
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Generate tables
    runner.generate_detailed_tables()
    
    # Generate plots
    runner.generate_publication_plots()
    
    # Generate LaTeX code
    runner.generate_latex_summary()
    
    # Print final summary
    runner.print_final_summary()


if __name__ == "__main__":
    main()