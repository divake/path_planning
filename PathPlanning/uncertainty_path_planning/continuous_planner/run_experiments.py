#!/usr/bin/env python3
"""
MAIN EXPERIMENT RUNNER FOR ICRA
Run comprehensive experiments comparing Naive vs Standard CP vs Learnable CP
This is the main entry point for generating paper-ready results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from typing import Dict, List, Tuple
import json

sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP, ContinuousNonconformity
from learnable_cp_final import FinalLearnableCP
from rrt_star_planner import RRTStar
from continuous_visualization import visualize_comparison, create_uncertainty_visualization
from proper_monte_carlo_evaluation import monte_carlo_evaluation_single_env
from cross_validation_environments import get_training_environments, get_test_environments


def run_comprehensive_experiments(num_trials: int = 500, 
                                 noise_level: float = 0.15,
                                 save_results: bool = True):
    """
    Run comprehensive experiments on all environments
    
    Args:
        num_trials: Number of Monte Carlo trials per environment
        noise_level: Perception noise level (0.15 = 15%)
        save_results: Whether to save results to file
    
    Returns:
        Dictionary with all results
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENTS FOR ICRA 2025")
    print("="*80)
    print(f"Settings: {num_trials} trials, {noise_level*100:.0f}% noise")
    
    # Get all environments
    all_envs = {}
    all_envs.update(get_training_environments())
    all_envs.update(get_test_environments())
    
    # Methods to evaluate
    methods = ['naive', 'standard', 'learnable']
    
    # Store all results
    all_results = {}
    
    # Train Learnable CP if needed
    if 'learnable' in methods:
        print("\nTraining Learnable CP model...")
        model = FinalLearnableCP(coverage=0.90, max_tau=1.0)
        
        # Set baseline taus for training
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
        
        # Train the model
        model.train(num_epochs=500, batch_size=32, learning_rate=0.001)
        print("Training complete!")
    
    # Run experiments for each environment
    for env_name in all_envs.keys():
        print(f"\n{'='*60}")
        print(f"ENVIRONMENT: {env_name.upper()}")
        print(f"{'='*60}")
        
        all_results[env_name] = {}
        
        for method in methods:
            print(f"\nEvaluating {method.upper()}...")
            
            result = monte_carlo_evaluation_single_env(
                env_type=env_name,
                num_trials=num_trials,
                noise_level=noise_level,
                method=method
            )
            
            all_results[env_name][method] = result
            
            # Print summary
            print(f"  Success rate: {result['success_rate']*100:.1f}%")
            print(f"  Collision rate: {result['collision_rate']*100:.1f}%")
            print(f"  Paths found: {result['paths_found']}/{num_trials}")
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Environment':<15} {'Method':<10} {'Success':<12} {'Collision':<12} {'Path Found':<12}")
    print("-"*80)
    
    for env_name in all_results:
        for method in all_results[env_name]:
            r = all_results[env_name][method]
            success = f"{r['success_rate']*100:.1f}%"
            collision = f"{r['collision_rate']*100:.1f}%"
            path_rate = f"{r['paths_found']}/{num_trials}"
            print(f"{env_name:<15} {method:<10} {success:<12} {collision:<12} {path_rate:<12}")
        print()  # Empty line between environments
    
    # Save results if requested
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/experiment_results_{timestamp}.json"
        
        # Create results directory if needed
        import os
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    return all_results


def generate_paper_figures(results: Dict):
    """
    Generate publication-quality figures from results
    
    Args:
        results: Dictionary with experiment results
    """
    
    print("\n" + "="*80)
    print("GENERATING PAPER FIGURES")
    print("="*80)
    
    # Create results directory
    import os
    os.makedirs("results/figures", exist_ok=True)
    
    # Figure 1: Success rates comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Success Rates Across Environments", fontsize=16)
    
    env_names = list(results.keys())[:6]  # First 6 environments
    
    for idx, env_name in enumerate(env_names):
        ax = axes[idx // 3, idx % 3]
        
        methods = ['naive', 'standard', 'learnable']
        success_rates = [results[env_name][m]['success_rate']*100 for m in methods]
        colors = ['red', 'blue', 'green']
        
        bars = ax.bar(methods, success_rates, color=colors, alpha=0.7)
        ax.set_title(f"{env_name.title()}", fontsize=12)
        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% target')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("results/figures/success_rates_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/success_rates_comparison.png", dpi=150, bbox_inches='tight')
    print("  Saved: success_rates_comparison.pdf/png")
    
    # Figure 2: Collision rates vs Path finding rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collision rates
    for method in ['naive', 'standard', 'learnable']:
        collision_rates = [results[env][method]['collision_rate']*100 
                          for env in env_names if method in results[env]]
        ax1.plot(range(len(collision_rates)), collision_rates, 
                marker='o', label=method.title(), linewidth=2)
    
    ax1.set_xlabel("Environment Index")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rates Across Environments")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(env_names)))
    ax1.set_xticklabels([e[:3].upper() for e in env_names], rotation=45)
    
    # Path finding rates
    for method in ['naive', 'standard', 'learnable']:
        total_trials = list(results.values())[0][method].get('paths_found', 0) + \
                      list(results.values())[0][method].get('collisions', 0)
        if total_trials == 0:
            total_trials = 500  # Default
        path_rates = [results[env][method]['paths_found']/total_trials*100 
                     for env in env_names if method in results[env]]
        ax2.plot(range(len(path_rates)), path_rates, 
                marker='s', label=method.title(), linewidth=2)
    
    ax2.set_xlabel("Environment Index")
    ax2.set_ylabel("Path Finding Rate (%)")
    ax2.set_title("Path Finding Success Across Environments")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(env_names)))
    ax2.set_xticklabels([e[:3].upper() for e in env_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/figures/collision_vs_pathfinding.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("results/figures/collision_vs_pathfinding.png", dpi=150, bbox_inches='tight')
    print("  Saved: collision_vs_pathfinding.pdf/png")
    
    print("\nAll figures saved to results/figures/")


def main():
    """
    Main entry point for experiments
    """
    
    # Run experiments
    results = run_comprehensive_experiments(
        num_trials=500,      # Use 500 for paper, 100 for testing
        noise_level=0.15,    # 15% perception noise
        save_results=True
    )
    
    # Generate figures
    generate_paper_figures(results)
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    
    # Calculate average success rates
    avg_success = {}
    for method in ['naive', 'standard', 'learnable']:
        rates = []
        for env in results:
            if method in results[env]:
                rates.append(results[env][method]['success_rate'])
        avg_success[method] = np.mean(rates) * 100
    
    print(f"\nAverage Success Rates:")
    print(f"  Naive:      {avg_success['naive']:.1f}%")
    print(f"  Standard:   {avg_success['standard']:.1f}%")
    print(f"  Learnable:  {avg_success['learnable']:.1f}%")
    
    # Check coverage guarantee
    coverage_met = {}
    for method in ['standard', 'learnable']:
        met_count = 0
        total_count = 0
        for env in results:
            if method in results[env]:
                total_count += 1
                if results[env][method]['ci_lower'] >= 0.90:
                    met_count += 1
        coverage_met[method] = (met_count, total_count)
    
    print(f"\nCoverage Guarantee (≥90%):")
    print(f"  Standard:   {coverage_met['standard'][0]}/{coverage_met['standard'][1]} environments")
    print(f"  Learnable:  {coverage_met['learnable'][0]}/{coverage_met['learnable'][1]} environments")
    
    print("\n✓ Ready for paper submission!")
    

if __name__ == "__main__":
    main()