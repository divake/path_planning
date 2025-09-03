#!/usr/bin/env python3
"""
Main Continuous Planner Comparison
Orchestrates all studies and generates comprehensive results
"""

import numpy as np
import time
from typing import Dict, List
import os
import json

from rrt_star_planner import RRTStar
from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from continuous_visualization import ContinuousVisualizer
from ablation_studies import AblationStudies


class ContinuousComparison:
    """
    Main comparison framework for continuous planning
    """
    
    def __init__(self):
        """Initialize comparison framework"""
        self.env = ContinuousEnvironment()
        self.visualizer = ContinuousVisualizer()
        self.ablation = AblationStudies(self.env)
        
        # Create results directory
        os.makedirs("continuous_planner/results", exist_ok=True)
        
    def run_naive_vs_cp_comparison(self, num_trials: int = 100):
        """
        Main comparison: Naive vs Standard CP
        
        Args:
            num_trials: Number of Monte Carlo trials
        """
        print("\n" + "="*70)
        print("MAIN COMPARISON: Naive vs Standard CP")
        print("="*70)
        
        # Initialize CP
        cp = ContinuousStandardCP(self.env.obstacles, "penetration")
        tau = cp.calibrate(
            ContinuousNoiseModel.add_thinning_noise,
            {'thin_factor': 0.2},
            num_samples=200,
            confidence=0.95
        )
        
        results = {
            'Naive': {
                'collisions': 0,
                'successes': 0,
                'path_lengths': [],
                'computation_times': [],
                'collision_points': []
            },
            'Standard CP': {
                'collisions': 0,
                'successes': 0,
                'path_lengths': [],
                'computation_times': [],
                'collision_points': []
            }
        }
        
        # Collect sample paths for visualization
        sample_results = {}
        sample_indices = [0, 10, 20, 30, 40, 50]  # Sample trials to visualize
        
        print(f"\nRunning {num_trials} trials...")
        print(f"τ = {tau:.3f}")
        
        for trial in range(num_trials):
            if trial % 20 == 0:
                print(f"  Trial {trial}/{num_trials}...")
            
            # Generate noisy perception
            perceived = ContinuousNoiseModel.add_thinning_noise(
                self.env.obstacles, thin_factor=0.2, seed=trial
            )
            
            # Naive planning
            start_time = time.time()
            naive_planner = RRTStar((5, 15), (45, 15), perceived, max_iter=500)
            naive_path = naive_planner.plan()
            naive_time = (time.time() - start_time) * 1000
            
            if naive_path:
                results['Naive']['successes'] += 1
                results['Naive']['path_lengths'].append(
                    naive_planner.get_metrics()['path_length']
                )
                results['Naive']['computation_times'].append(naive_time)
                
                # Check collisions
                collision_points = []
                for p in naive_path:
                    if self.env.point_in_obstacle(p[0], p[1]):
                        collision_points.append(p)
                
                if collision_points:
                    results['Naive']['collisions'] += 1
                    results['Naive']['collision_points'].extend(collision_points)
            
            # CP planning
            start_time = time.time()
            inflated = cp.inflate_obstacles(perceived)
            cp_planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
            cp_path = cp_planner.plan()
            cp_time = (time.time() - start_time) * 1000
            
            if cp_path:
                results['Standard CP']['successes'] += 1
                results['Standard CP']['path_lengths'].append(
                    cp_planner.get_metrics()['path_length']
                )
                results['Standard CP']['computation_times'].append(cp_time)
                
                # Check collisions
                collision_points = []
                for p in cp_path:
                    if self.env.point_in_obstacle(p[0], p[1]):
                        collision_points.append(p)
                
                if collision_points:
                    results['Standard CP']['collisions'] += 1
                    results['Standard CP']['collision_points'].extend(collision_points)
            
            # Store sample results for visualization
            if trial in sample_indices:
                sample_results[f'Trial {trial} - Naive'] = {
                    'path': naive_path,
                    'perceived_obs': perceived,
                    'collisions': collision_points if naive_path else [],
                    'metrics': {
                        'path_length': naive_planner.get_metrics()['path_length'] if naive_path else 0,
                        'num_collisions': len(collision_points) if naive_path else 0
                    }
                }
                
                sample_results[f'Trial {trial} - CP'] = {
                    'path': cp_path,
                    'perceived_obs': perceived,
                    'inflated_obs': inflated,
                    'collisions': collision_points if cp_path else [],
                    'metrics': {
                        'path_length': cp_planner.get_metrics()['path_length'] if cp_path else 0,
                        'num_collisions': len(collision_points) if cp_path else 0
                    }
                }
        
        # Calculate statistics
        for method in results:
            if results[method]['successes'] > 0:
                collision_rate = (results[method]['collisions'] / 
                                results[method]['successes'] * 100)
                avg_path_length = np.mean(results[method]['path_lengths'])
                avg_time = np.mean(results[method]['computation_times'])
            else:
                collision_rate = 0
                avg_path_length = 0
                avg_time = 0
            
            print(f"\n{method} Results:")
            print(f"  Success rate: {results[method]['successes']}/{num_trials} "
                  f"({results[method]['successes']/num_trials*100:.1f}%)")
            print(f"  Collision rate: {collision_rate:.2f}%")
            print(f"  Avg path length: {avg_path_length:.2f}")
            print(f"  Avg computation time: {avg_time:.2f} ms")
        
        # Visualize sample results
        self.visualizer.plot_comparison(sample_results, 
                                       "Naive vs Standard CP Comparison")
        
        return results
    
    def run_all_ablation_studies(self):
        """Run all ablation studies"""
        print("\n" + "="*70)
        print("RUNNING ALL ABLATION STUDIES")
        print("="*70)
        
        all_results = {}
        
        # 1. Noise level study
        print("\n1. Noise Level Study")
        noise_results = self.ablation.study_noise_levels(
            noise_levels=[0.1, 0.15, 0.2, 0.25, 0.3],
            num_trials=50
        )
        all_results['noise_levels'] = noise_results
        
        # 2. Nonconformity score comparison
        print("\n2. Nonconformity Score Comparison")
        score_results = self.ablation.study_nonconformity_scores(num_trials=100)
        all_results['nonconformity_scores'] = score_results
        
        # 3. Noise model comparison
        print("\n3. Noise Model Comparison")
        model_results = self.ablation.study_noise_models(num_trials=50)
        all_results['noise_models'] = model_results
        
        # 4. Computation time study
        print("\n4. Computation Time Study")
        time_results = self.ablation.study_computation_time(num_trials=30)
        all_results['computation_times'] = time_results
        
        # 5. Coverage guarantee validation
        print("\n5. Coverage Guarantee Validation")
        coverage_results = self.ablation.validate_coverage_guarantee(
            confidence_levels=[0.80, 0.90, 0.95, 0.99],
            num_trials=200
        )
        all_results['coverage_validation'] = coverage_results
        
        # Create summary
        summary = self._create_summary(all_results)
        all_results['summary'] = summary
        
        # Visualize ablation results
        self.visualizer.plot_ablation_results(all_results)
        
        # Save results to JSON
        self._save_results(all_results)
        
        return all_results
    
    def run_tau_analysis(self):
        """Analyze τ values and coverage"""
        print("\n" + "="*70)
        print("TAU (τ) ANALYSIS")
        print("="*70)
        
        tau_curves = {}
        coverage_data = {}
        
        # Test different nonconformity scores
        score_types = ["penetration", "hausdorff", "area"]
        
        for score_type in score_types:
            print(f"\nAnalyzing {score_type} score...")
            
            cp = ContinuousStandardCP(self.env.obstacles, score_type)
            
            # Get τ curve for different confidence levels
            cp.calibrate(
                ContinuousNoiseModel.add_thinning_noise,
                {'thin_factor': 0.2},
                num_samples=500,
                confidence=0.95
            )
            
            tau_curve = cp.get_tau_curve()
            tau_curves[score_type] = tau_curve
            
            # Test coverage at different τ values
            tau_values = sorted(set(cp.calibration_scores))[:10]
            coverages = []
            
            for tau_val in tau_values:
                coverage = cp.compute_coverage(tau_val)
                coverages.append(coverage * 100)
            
            coverage_data[score_type] = {
                'tau_values': tau_values,
                'coverages': coverages
            }
        
        # Store score distributions
        coverage_data['score_distributions'] = {
            score_type: cp.calibration_scores 
            for score_type in score_types
        }
        
        # Create comparison table
        table_data = [
            ['Score Type', 'τ@95%', 'Mean', 'Std', 'Range']
        ]
        
        for score_type in score_types:
            cp = ContinuousStandardCP(self.env.obstacles, score_type)
            cp.calibrate(
                ContinuousNoiseModel.add_thinning_noise,
                {'thin_factor': 0.2},
                num_samples=200,
                confidence=0.95
            )
            
            scores = cp.calibration_scores
            table_data.append([
                score_type,
                f'{cp.tau:.3f}',
                f'{np.mean(scores):.3f}',
                f'{np.std(scores):.3f}',
                f'[{min(scores):.2f}, {max(scores):.2f}]'
            ])
        
        coverage_data['comparison_table'] = table_data
        
        # Visualize τ analysis
        self.visualizer.plot_tau_analysis(tau_curves, coverage_data)
        
        return tau_curves, coverage_data
    
    def run_noise_effect_visualization(self):
        """Visualize effects of different noise models"""
        print("\n" + "="*70)
        print("NOISE MODEL EFFECTS VISUALIZATION")
        print("="*70)
        
        noise_results = {}
        
        noise_configs = [
            ("True Obstacles", None, None),
            ("Gaussian σ=0.1", ContinuousNoiseModel.add_gaussian_noise, {'noise_std': 0.1}),
            ("Gaussian σ=0.3", ContinuousNoiseModel.add_gaussian_noise, {'noise_std': 0.3}),
            ("Thinning 10%", ContinuousNoiseModel.add_thinning_noise, {'thin_factor': 0.1}),
            ("Thinning 30%", ContinuousNoiseModel.add_thinning_noise, {'thin_factor': 0.3}),
            ("Expansion 10%", ContinuousNoiseModel.add_expansion_noise, {'expand_factor': 0.1}),
            ("Expansion 30%", ContinuousNoiseModel.add_expansion_noise, {'expand_factor': 0.3}),
            ("Mixed Model", ContinuousNoiseModel.add_mixed_noise, 
             {'gaussian_std': 0.1, 'thin_prob': 0.2, 'expand_prob': 0.2})
        ]
        
        for name, noise_func, params in noise_configs:
            if noise_func is None:
                # True obstacles
                perceived = self.env.obstacles
                score = 0
            else:
                perceived = noise_func(self.env.obstacles, **params, seed=42)
                
                # Compute nonconformity score
                cp = ContinuousStandardCP(self.env.obstacles, "penetration")
                from continuous_standard_cp import ContinuousNonconformity
                score = ContinuousNonconformity.compute_penetration_depth(
                    self.env.obstacles, perceived
                )
            
            noise_results[name] = {
                'true_obs': self.env.obstacles,
                'perceived_obs': perceived,
                'stats': {'score': score}
            }
        
        # Visualize noise effects
        self.visualizer.plot_noise_effects(noise_results)
        
        return noise_results
    
    def _create_summary(self, results: Dict) -> str:
        """Create summary text for results"""
        summary = "ABLATION STUDY SUMMARY\n"
        summary += "="*30 + "\n\n"
        
        # Noise level insights
        if 'noise_levels' in results:
            naive_collisions = results['noise_levels']['naive']['collision_rates']
            cp_collisions = results['noise_levels']['standard_cp']['collision_rates']
            
            summary += "Noise Level Impact:\n"
            summary += f"• Naive: {min(naive_collisions):.1f}% → {max(naive_collisions):.1f}%\n"
            summary += f"• CP: {min(cp_collisions):.1f}% → {max(cp_collisions):.1f}%\n"
            summary += f"• CP reduces collisions by {np.mean(naive_collisions) - np.mean(cp_collisions):.1f}%\n\n"
        
        # Nonconformity score insights
        if 'nonconformity_scores' in results:
            summary += "Best Nonconformity Score:\n"
            best_score = min(results['nonconformity_scores'].items(),
                           key=lambda x: x[1]['collision_rate'])
            summary += f"• {best_score[0]}: {best_score[1]['collision_rate']:.1f}% collisions\n"
            summary += f"• τ = {best_score[1]['tau']:.3f}\n\n"
        
        # Computation time
        if 'computation_times' in results:
            summary += "Computation Overhead:\n"
            naive_time = results['computation_times']['Naive']['mean_time']
            cp_time = results['computation_times']['CP-Uniform']['mean_time']
            overhead = (cp_time - naive_time) / naive_time * 100
            summary += f"• CP adds {overhead:.1f}% overhead\n"
            summary += f"• {cp_time - naive_time:.1f} ms extra\n\n"
        
        # Coverage validation
        if 'coverage_validation' in results:
            summary += "Coverage Guarantees:\n"
            for conf, data in results['coverage_validation'].items():
                summary += f"• {conf}: {data['actual']:.1f}% actual\n"
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open('continuous_planner/results/ablation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("\nResults saved to continuous_planner/results/ablation_results.json")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("CONTINUOUS PLANNER COMPREHENSIVE STUDY")
    print("Conformal Prediction for Path Planning")
    print("="*70)
    
    comparison = ContinuousComparison()
    
    # 1. Main comparison
    print("\n[1/4] Running main Naive vs CP comparison...")
    main_results = comparison.run_naive_vs_cp_comparison(num_trials=200)
    
    # 2. τ analysis
    print("\n[2/4] Running τ analysis...")
    tau_curves, coverage_data = comparison.run_tau_analysis()
    
    # 3. Noise effects
    print("\n[3/4] Visualizing noise model effects...")
    noise_results = comparison.run_noise_effect_visualization()
    
    # 4. Ablation studies
    print("\n[4/4] Running comprehensive ablation studies...")
    ablation_results = comparison.run_all_ablation_studies()
    
    print("\n" + "="*70)
    print("ALL STUDIES COMPLETE!")
    print("Results saved to continuous_planner/results/")
    print("="*70)
    
    # Print final summary
    print("\nFINAL SUMMARY:")
    print("-"*30)
    
    # Main comparison summary
    naive_collision_rate = (main_results['Naive']['collisions'] / 
                           main_results['Naive']['successes'] * 100 
                           if main_results['Naive']['successes'] > 0 else 0)
    cp_collision_rate = (main_results['Standard CP']['collisions'] / 
                        main_results['Standard CP']['successes'] * 100 
                        if main_results['Standard CP']['successes'] > 0 else 0)
    
    print(f"Naive Method: {naive_collision_rate:.2f}% collision rate")
    print(f"Standard CP: {cp_collision_rate:.2f}% collision rate")
    print(f"Improvement: {naive_collision_rate - cp_collision_rate:.2f}% reduction")
    
    print("\nKey Findings:")
    print("• Continuous planners allow fine-grained τ values")
    print("• Standard CP provides statistical safety guarantees")
    print("• Trade-off between safety and path optimality")
    print("• Computational overhead is acceptable for safety gains")


if __name__ == "__main__":
    main()