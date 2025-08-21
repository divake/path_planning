#!/usr/bin/env python3
"""
Full Experiment Runner - Runs all three methods on diverse scenarios
Generates comprehensive results and figures for ICRA paper
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import json
import pickle
from datetime import datetime
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icra_implementation.memory_system import MemorySystem
from icra_implementation.collision_checker import CollisionChecker
from icra_implementation.methods.naive_planner import NaivePlanner
from icra_implementation.methods.ensemble_planner import EnsemblePlanner
from icra_implementation.methods.learnable_cp_planner import LearnableConformalPlanner

# Initialize memory system
memory = MemorySystem()
memory.log_progress("FULL_EXPERIMENT", "STARTED", "Beginning comprehensive experiments")

class FullExperimentRunner:
    def __init__(self):
        self.memory = memory
        self.collision_checker = CollisionChecker()
        
        # Initialize planners
        self.naive_planner = NaivePlanner(memory_system=memory)
        self.ensemble_planner = EnsemblePlanner(n_models=5, memory_system=memory)
        self.cp_planner = LearnableConformalPlanner(alpha=0.05, memory_system=memory)
        
        # Results storage
        self.all_results = []
        
    def generate_comprehensive_scenarios(self):
        """Generate diverse scenarios for comprehensive evaluation"""
        scenarios = []
        scenario_id = 0
        
        # Different environment types
        environments = [
            {
                'name': 'sparse',
                'n_obstacles': 5,
                'obstacle_radius': (0.5, 1.5),
                'area': (30, 30),
                'count': 20
            },
            {
                'name': 'moderate',
                'n_obstacles': 10,
                'obstacle_radius': (0.5, 2.0),
                'area': (30, 30),
                'count': 30
            },
            {
                'name': 'dense',
                'n_obstacles': 15,
                'obstacle_radius': (0.5, 2.5),
                'area': (30, 30),
                'count': 30
            },
            {
                'name': 'narrow_passage',
                'n_obstacles': 8,
                'obstacle_radius': (1.0, 2.0),
                'area': (30, 30),
                'count': 20,
                'special': 'narrow_passage'
            }
        ]
        
        for env in environments:
            for i in range(env['count']):
                # Random start and goal
                start = [
                    np.random.uniform(0, 5),
                    np.random.uniform(0, 5),
                    np.random.uniform(0, 2*np.pi)
                ]
                
                goal = [
                    np.random.uniform(25, env['area'][0]),
                    np.random.uniform(25, env['area'][1]),
                    np.random.uniform(0, 2*np.pi)
                ]
                
                # Generate obstacles
                obstacles = []
                
                if env.get('special') == 'narrow_passage':
                    # Create a narrow passage scenario
                    # Wall of obstacles with a gap
                    passage_y = env['area'][1] / 2
                    gap_x = env['area'][0] / 2
                    gap_width = np.random.uniform(3, 5)
                    
                    for x in np.linspace(5, 25, 8):
                        if abs(x - gap_x) > gap_width / 2:
                            obstacles.append([
                                x,
                                passage_y + np.random.uniform(-1, 1),
                                np.random.uniform(1.0, 2.0)
                            ])
                else:
                    # Random obstacles
                    for _ in range(env['n_obstacles']):
                        attempts = 0
                        while attempts < 10:
                            x = np.random.uniform(5, 25)
                            y = np.random.uniform(5, 25)
                            radius = np.random.uniform(*env['obstacle_radius'])
                            
                            # Check not too close to start/goal
                            if (np.sqrt((x - start[0])**2 + (y - start[1])**2) > radius + 2 and
                                np.sqrt((x - goal[0])**2 + (y - goal[1])**2) > radius + 2):
                                obstacles.append([x, y, radius])
                                break
                            attempts += 1
                
                scenario = {
                    'id': scenario_id,
                    'environment': env['name'],
                    'start': start,
                    'goal': goal,
                    'obstacles': obstacles,
                    'noise_level': np.random.uniform(0.1, 0.5)
                }
                
                scenarios.append(scenario)
                scenario_id += 1
        
        # Save scenarios
        os.makedirs('icra_implementation/data', exist_ok=True)
        with open('icra_implementation/data/comprehensive_scenarios.pkl', 'wb') as f:
            pickle.dump(scenarios, f)
        
        self.memory.log_progress("SCENARIO_GENERATION", "COMPLETED", 
                                f"Generated {len(scenarios)} diverse scenarios")
        
        return scenarios
    
    def prepare_training_data(self, scenarios):
        """Prepare training data for learnable CP"""
        training_data = []
        
        # Use first 30% of scenarios for training
        n_train = int(len(scenarios) * 0.3)
        train_scenarios = scenarios[:n_train]
        
        self.memory.log_progress("TRAINING_DATA", "STARTED", 
                                f"Preparing training data from {n_train} scenarios")
        
        for scenario in train_scenarios:
            # Run naive planner to get paths
            path, metrics = self.naive_planner.plan(
                scenario['start'],
                scenario['goal'],
                scenario['obstacles']
            )
            
            if path:
                # Convert path to training format
                path_states = [(p.x, p.y, p.yaw) for p in path.x_list]
                
                # Simulate tracking errors (for training nonconformity)
                tracking_errors = []
                for i, state in enumerate(path_states):
                    # Error increases near obstacles
                    min_dist = min([np.sqrt((state[0] - obs[0])**2 + 
                                          (state[1] - obs[1])**2) - obs[2]
                                  for obs in scenario['obstacles']])
                    
                    # Error model: higher near obstacles
                    base_error = 0.1
                    proximity_error = max(0, 1.0 - min_dist / 5.0)
                    noise = np.random.normal(0, 0.05)
                    
                    error = base_error + proximity_error + noise
                    tracking_errors.append(max(0, error))
                
                training_data.append({
                    'start': scenario['start'],
                    'goal': scenario['goal'],
                    'obstacles': scenario['obstacles'],
                    'path': path_states,
                    'tracking_errors': tracking_errors
                })
        
        self.memory.log_progress("TRAINING_DATA", "COMPLETED", 
                                f"Prepared {len(training_data)} training samples")
        
        return training_data
    
    def run_comprehensive_experiments(self):
        """Run all experiments"""
        
        # Generate scenarios
        scenarios = self.generate_comprehensive_scenarios()
        
        # Prepare training data and train CP model
        training_data = self.prepare_training_data(scenarios)
        
        # Train learnable CP
        self.memory.log_progress("CP_TRAINING", "STARTED", "Training learnable CP model")
        self.cp_planner.train(training_data)
        
        # Calibrate CP model
        calibration_scenarios = scenarios[int(len(scenarios)*0.3):int(len(scenarios)*0.4)]
        calibration_data = self.prepare_training_data(calibration_scenarios)
        self.cp_planner.calibrate(calibration_data)
        
        # Save trained model
        self.cp_planner.save_model('icra_implementation/checkpoints/cp_model.pth')
        
        # Test scenarios (last 60%)
        test_scenarios = scenarios[int(len(scenarios)*0.4):]
        
        self.memory.log_progress("EVALUATION", "STARTED", 
                                f"Evaluating on {len(test_scenarios)} test scenarios")
        
        # Run experiments
        for i, scenario in enumerate(test_scenarios):
            if i % 10 == 0:
                self.memory.log_progress("EVALUATION", "PROGRESS", 
                                       f"Processing scenario {i}/{len(test_scenarios)}")
            
            # Run naive planner
            naive_path, naive_metrics = self.naive_planner.plan(
                scenario['start'],
                scenario['goal'],
                scenario['obstacles']
            )
            
            # Run ensemble planner
            ensemble_path, ensemble_metrics = self.ensemble_planner.plan(
                scenario['start'],
                scenario['goal'],
                scenario['obstacles'],
                noise_level=scenario['noise_level']
            )
            
            # Run learnable CP planner
            cp_path, cp_metrics = self.cp_planner.plan(
                scenario['start'],
                scenario['goal'],
                scenario['obstacles']
            )
            
            # Store results
            result = {
                'scenario_id': scenario['id'],
                'environment': scenario['environment'],
                'naive': naive_metrics,
                'ensemble': ensemble_metrics,
                'learnable_cp': cp_metrics
            }
            
            self.all_results.append(result)
            
            # Save intermediate results
            if i % 20 == 0:
                self.save_results()
        
        self.memory.log_progress("EVALUATION", "COMPLETED", 
                                f"Completed evaluation on all test scenarios")
        
        # Generate comprehensive figures
        self.generate_comprehensive_figures()
        
        # Create summary report
        self.create_detailed_report()
        
        return self.all_results
    
    def save_results(self):
        """Save results to files"""
        # Save as JSON
        with open('icra_implementation/results/comprehensive_results.json', 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Save as pickle for later analysis
        with open('icra_implementation/results/comprehensive_results.pkl', 'wb') as f:
            pickle.dump(self.all_results, f)
        
        # Convert to CSV for analysis
        rows = []
        for result in self.all_results:
            for method in ['naive', 'ensemble', 'learnable_cp']:
                if method in result and result[method]:
                    row = {
                        'scenario_id': result['scenario_id'],
                        'environment': result['environment'],
                        'method': method,
                        **result[method]
                    }
                    rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv('icra_implementation/results/comprehensive_results.csv', index=False)
    
    def generate_comprehensive_figures(self):
        """Generate all publication figures"""
        
        self.memory.log_progress("FIGURES", "STARTED", "Generating publication figures")
        
        # 1. Safety-Performance Tradeoff
        self.plot_safety_performance_tradeoff()
        
        # 2. Environment-specific performance
        self.plot_environment_performance()
        
        # 3. Uncertainty visualization
        self.plot_uncertainty_adaptation()
        
        # 4. Coverage analysis
        self.plot_coverage_analysis()
        
        # 5. Computational efficiency
        self.plot_computational_efficiency()
        
        self.memory.log_progress("FIGURES", "COMPLETED", "All figures generated")
    
    def plot_safety_performance_tradeoff(self):
        """Plot safety vs performance tradeoff"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['naive', 'ensemble', 'learnable_cp']
        colors = ['red', 'blue', 'green']
        markers = ['o', 's', '^']
        
        for method, color, marker in zip(methods, colors, markers):
            collision_rates = []
            path_lengths = []
            
            for result in self.all_results:
                if method in result and result[method]:
                    collision_rates.append(result[method].get('collision_rate', 0))
                    path_lengths.append(result[method].get('path_length', 0))
            
            if collision_rates and path_lengths:
                # Calculate averages for each environment type
                env_types = set(r['environment'] for r in self.all_results)
                
                for env in env_types:
                    env_collision = []
                    env_length = []
                    
                    for result in self.all_results:
                        if result['environment'] == env and method in result and result[method]:
                            env_collision.append(result[method].get('collision_rate', 0))
                            env_length.append(result[method].get('path_length', 0))
                    
                    if env_collision and env_length:
                        avg_collision = np.mean(env_collision)
                        avg_length = np.mean(env_length)
                        
                        ax.scatter(avg_length, 1 - avg_collision, 
                                 color=color, marker=marker, s=150, alpha=0.7,
                                 label=f'{method.replace("_", " ").title()} ({env})')
        
        ax.set_xlabel('Average Path Length (m)', fontsize=14)
        ax.set_ylabel('Safety (1 - Collision Rate)', fontsize=14)
        ax.set_title('Safety-Performance Tradeoff Across Methods', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('icra_implementation/figures/safety_performance_tradeoff.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_environment_performance(self):
        """Plot performance across different environments"""
        env_types = list(set(r['environment'] for r in self.all_results))
        methods = ['naive', 'ensemble', 'learnable_cp']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_to_plot = [
            ('collision_rate', 'Collision Rate'),
            ('path_length', 'Path Length (m)'),
            ('min_clearance', 'Minimum Clearance (m)'),
            ('planning_time', 'Planning Time (s)')
        ]
        
        for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            data_by_method = {method: {env: [] for env in env_types} for method in methods}
            
            for result in self.all_results:
                env = result['environment']
                for method in methods:
                    if method in result and result[method]:
                        value = result[method].get(metric_key, 0)
                        data_by_method[method][env].append(value)
            
            # Calculate means and stds
            x = np.arange(len(env_types))
            width = 0.25
            
            for i, method in enumerate(methods):
                means = [np.mean(data_by_method[method][env]) if data_by_method[method][env] else 0 
                        for env in env_types]
                stds = [np.std(data_by_method[method][env]) if data_by_method[method][env] else 0 
                       for env in env_types]
                
                ax.bar(x + i * width, means, width, 
                      yerr=stds, capsize=5,
                      label=method.replace('_', ' ').title())
            
            ax.set_xlabel('Environment Type', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} by Environment', fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(env_types)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('icra_implementation/figures/environment_performance.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_uncertainty_adaptation(self):
        """Visualize how uncertainty adapts to different scenarios"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Select representative scenarios
        env_types = ['sparse', 'moderate', 'dense']
        
        for idx, env_type in enumerate(env_types):
            ax = axes[idx]
            
            # Find a scenario of this type
            scenario = None
            cp_result = None
            
            for result in self.all_results:
                if result['environment'] == env_type and 'learnable_cp' in result:
                    scenario_id = result['scenario_id']
                    cp_result = result['learnable_cp']
                    
                    # Load the scenario
                    with open('icra_implementation/data/comprehensive_scenarios.pkl', 'rb') as f:
                        all_scenarios = pickle.load(f)
                        for s in all_scenarios:
                            if s['id'] == scenario_id:
                                scenario = s
                                break
                    break
            
            if scenario and cp_result:
                # Plot obstacles
                for obs in scenario['obstacles']:
                    circle = plt.Circle((obs[0], obs[1]), obs[2], 
                                       color='gray', alpha=0.5)
                    ax.add_patch(circle)
                
                # Plot start and goal
                ax.plot(scenario['start'][0], scenario['start'][1], 
                       'go', markersize=10, label='Start')
                ax.plot(scenario['goal'][0], scenario['goal'][1], 
                       'r*', markersize=15, label='Goal')
                
                # Color code by uncertainty if available
                uncertainty = cp_result.get('uncertainty_efficiency', 0)
                adaptivity = cp_result.get('adaptivity_score', 0)
                
                ax.set_title(f'{env_type.title()} Environment\n'
                           f'Uncertainty: {uncertainty:.3f}, Adaptivity: {adaptivity:.3f}')
                
            ax.set_xlim(-5, 35)
            ax.set_ylim(-5, 35)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('icra_implementation/figures/uncertainty_adaptation.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_coverage_analysis(self):
        """Analyze coverage rates for CP method"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract coverage data for CP method
        cp_coverage = []
        cp_efficiency = []
        cp_adaptivity = []
        
        for result in self.all_results:
            if 'learnable_cp' in result and result['learnable_cp']:
                cp_coverage.append(result['learnable_cp'].get('coverage_rate', 0))
                cp_efficiency.append(result['learnable_cp'].get('uncertainty_efficiency', 0))
                cp_adaptivity.append(result['learnable_cp'].get('adaptivity_score', 0))
        
        # Plot 1: Coverage histogram
        ax1 = axes[0]
        if cp_coverage:
            ax1.hist(cp_coverage, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax1.axvline(0.95, color='red', linestyle='--', linewidth=2, label='Target (95%)')
            ax1.axvline(np.mean(cp_coverage), color='blue', linestyle='-', linewidth=2, 
                       label=f'Mean ({np.mean(cp_coverage):.3f})')
        
        ax1.set_xlabel('Coverage Rate', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Coverage Rate Distribution (Learnable CP)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency vs Adaptivity
        ax2 = axes[1]
        if cp_efficiency and cp_adaptivity:
            scatter = ax2.scatter(cp_efficiency, cp_adaptivity, 
                                c=cp_coverage, cmap='viridis', 
                                s=50, alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label='Coverage Rate')
        
        ax2.set_xlabel('Uncertainty Efficiency', fontsize=12)
        ax2.set_ylabel('Adaptivity Score', fontsize=12)
        ax2.set_title('Uncertainty Characteristics', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('icra_implementation/figures/coverage_analysis.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_computational_efficiency(self):
        """Plot computational efficiency comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['naive', 'ensemble', 'learnable_cp']
        colors = ['red', 'blue', 'green']
        
        planning_times = {method: [] for method in methods}
        success_rates = {method: [] for method in methods}
        
        for result in self.all_results:
            for method in methods:
                if method in result and result[method]:
                    planning_times[method].append(result[method].get('planning_time', 0))
                    success_rates[method].append(1 if result[method].get('success', False) else 0)
        
        # Create box plot
        data_to_plot = [planning_times[method] for method in methods]
        positions = np.arange(len(methods))
        
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, labels=[m.replace('_', ' ').title() for m in methods])
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        # Add success rates as text
        for i, method in enumerate(methods):
            if success_rates[method]:
                avg_success = np.mean(success_rates[method])
                ax.text(i, ax.get_ylim()[1] * 0.95, f'Success: {avg_success:.1%}',
                       ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Planning Time (seconds)', fontsize=14)
        ax.set_title('Computational Efficiency Comparison', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('icra_implementation/figures/computational_efficiency.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_report(self):
        """Create comprehensive report with all statistics"""
        report = []
        report.append("=" * 80)
        report.append("ICRA COMPREHENSIVE EXPERIMENT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Scenarios Evaluated: {len(self.all_results)}")
        report.append("")
        
        # Overall statistics
        methods = ['naive', 'ensemble', 'learnable_cp']
        
        report.append("OVERALL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for method in methods:
            metrics = {
                'collision_rate': [],
                'success_rate': [],
                'path_length': [],
                'min_clearance': [],
                'planning_time': []
            }
            
            for result in self.all_results:
                if method in result and result[method]:
                    metrics['collision_rate'].append(result[method].get('collision_rate', 0))
                    metrics['success_rate'].append(1 if result[method].get('success', False) else 0)
                    metrics['path_length'].append(result[method].get('path_length', 0))
                    metrics['min_clearance'].append(result[method].get('min_clearance', 0))
                    metrics['planning_time'].append(result[method].get('planning_time', 0))
            
            report.append(f"\n{method.upper().replace('_', ' ')}:")
            for metric_name, values in metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    report.append(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Environment-specific performance
        report.append("\n" + "=" * 40)
        report.append("ENVIRONMENT-SPECIFIC PERFORMANCE")
        report.append("-" * 40)
        
        env_types = list(set(r['environment'] for r in self.all_results))
        
        for env in env_types:
            report.append(f"\nEnvironment: {env.upper()}")
            
            for method in methods:
                collision_rates = []
                success_rates = []
                
                for result in self.all_results:
                    if result['environment'] == env and method in result and result[method]:
                        collision_rates.append(result[method].get('collision_rate', 0))
                        success_rates.append(1 if result[method].get('success', False) else 0)
                
                if collision_rates:
                    report.append(f"  {method}: Collision={np.mean(collision_rates):.3f}, "
                                f"Success={np.mean(success_rates):.1%}")
        
        # Special analysis for Learnable CP
        report.append("\n" + "=" * 40)
        report.append("LEARNABLE CP SPECIAL ANALYSIS")
        report.append("-" * 40)
        
        cp_coverage = []
        cp_efficiency = []
        cp_adaptivity = []
        
        for result in self.all_results:
            if 'learnable_cp' in result and result['learnable_cp']:
                cp_coverage.append(result['learnable_cp'].get('coverage_rate', 0))
                cp_efficiency.append(result['learnable_cp'].get('uncertainty_efficiency', 0))
                cp_adaptivity.append(result['learnable_cp'].get('adaptivity_score', 0))
        
        if cp_coverage:
            report.append(f"Average Coverage Rate: {np.mean(cp_coverage):.3f}")
            report.append(f"Coverage Std Dev: {np.std(cp_coverage):.3f}")
            report.append(f"Average Uncertainty Efficiency: {np.mean(cp_efficiency):.3f}")
            report.append(f"Average Adaptivity Score: {np.mean(cp_adaptivity):.3f}")
        
        report.append("\n" + "=" * 80)
        report.append("KEY FINDINGS:")
        report.append("-" * 40)
        
        # Calculate improvements
        naive_collision = np.mean([r['naive'].get('collision_rate', 0) 
                                  for r in self.all_results if 'naive' in r and r['naive']])
        cp_collision = np.mean([r['learnable_cp'].get('collision_rate', 0) 
                               for r in self.all_results if 'learnable_cp' in r and r['learnable_cp']])
        
        if naive_collision > 0:
            improvement = (naive_collision - cp_collision) / naive_collision * 100
            report.append(f"1. Learnable CP reduces collision rate by {improvement:.1f}% vs Naive")
        
        naive_length = np.mean([r['naive'].get('path_length', 0) 
                               for r in self.all_results if 'naive' in r and r['naive']])
        cp_length = np.mean([r['learnable_cp'].get('path_length', 0) 
                           for r in self.all_results if 'learnable_cp' in r and r['learnable_cp']])
        
        if naive_length > 0:
            length_increase = (cp_length - naive_length) / naive_length * 100
            report.append(f"2. Learnable CP paths are {length_increase:.1f}% longer (safety tradeoff)")
        
        report.append(f"3. Learnable CP achieves {np.mean(cp_coverage):.1%} coverage rate")
        report.append(f"4. Adaptive uncertainty shows {np.mean(cp_adaptivity):.3f} adaptivity score")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        with open('icra_implementation/COMPREHENSIVE_REPORT.txt', 'w') as f:
            f.write('\n'.join(report))
        
        # Also save as markdown
        with open('icra_implementation/COMPREHENSIVE_REPORT.md', 'w') as f:
            f.write('\n'.join(report))
        
        self.memory.log_progress("REPORT", "COMPLETED", "Comprehensive report generated")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Naive Collision Rate: {naive_collision:.3f}")
        print(f"Learnable CP Collision Rate: {cp_collision:.3f}")
        print(f"Coverage Rate: {np.mean(cp_coverage):.3f}")
        print(f"Results saved to icra_implementation/")
        print("=" * 60)

if __name__ == "__main__":
    runner = FullExperimentRunner()
    results = runner.run_comprehensive_experiments()
    
    memory.log_progress("FULL_EXPERIMENT", "COMPLETED", 
                       f"All experiments completed. Processed {len(results)} scenarios.")