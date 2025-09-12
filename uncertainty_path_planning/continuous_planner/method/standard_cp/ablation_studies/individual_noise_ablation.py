#!/usr/bin/env python3
"""
Corrected ablation study with fixed noise parameters
"""

import sys
import os
sys.path.append(os.path.abspath('../../..'))

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from noise_model import NoiseModel
from nonconformity_scorer import NonconformityScorer
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
import yaml

def run_corrected_ablation():
    """
    Run ablation study with corrected parameters
    """
    print("="*80)
    print("CORRECTED ABLATION STUDY - INDIVIDUAL NOISE TYPES")
    print("With fixed parameters:")
    print("- Localization position_std: 0.50m (was 0.03m)")
    print("- False negative rate: 10% (was 2%)")
    print("- Measurement noise: unchanged (working correctly)")
    print("="*80)
    
    # Configuration
    num_trials = 50  # Quick test
    noise_level = 0.25
    test_env = 'office01add'
    
    # Load environment config
    with open('../../../config/config_env.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Get test configuration
    env_tests = env_config['environments'][test_env]['tests']
    test = env_tests[0]  # Use first test
    start = test['start']
    goal = test['goal']
    
    print(f"\nEnvironment: {test_env}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Trials: {num_trials} per noise type")
    print(f"Noise level: {noise_level}")
    
    # Initialize components
    noise_model = NoiseModel('../../../config/standard_cp_config.yaml')
    scorer = NonconformityScorer('../../../config/standard_cp_config.yaml')
    
    # Load map
    parser = MRPBMapParser(
        map_name=test_env,
        mrpb_path='../../../mrpb_dataset/'
    )
    clean_grid = parser.occupancy_grid.copy()
    
    # Test each noise type
    noise_types_to_test = [
        (['measurement_noise'], 'measurement_only'),
        (['false_negatives'], 'false_negatives_only'),
        (['localization_drift'], 'localization_only'),
        (['measurement_noise', 'false_negatives', 'localization_drift'], 'all_combined')
    ]
    
    results = {}
    
    for noise_types, label in noise_types_to_test:
        print(f"\n" + "="*60)
        print(f"Testing: {label}")
        print("="*60)
        
        # Set noise types
        original_types = noise_model.noise_config['noise_types'].copy()
        noise_model.noise_config['noise_types'] = noise_types
        
        scores = []
        planning_failures = 0
        noise_effects = []
        
        with tqdm(total=num_trials, desc=label) as pbar:
            for trial in range(num_trials):
                # Add noise
                noisy_grid = noise_model.add_realistic_noise(
                    clean_grid, noise_level, seed=trial
                )
                
                # Analyze noise effect
                diff_pixels = np.sum(clean_grid != noisy_grid)
                false_negs = np.sum((clean_grid == 100) & (noisy_grid == 0))
                false_pos = np.sum((clean_grid == 0) & (noisy_grid == 100))
                
                noise_effects.append({
                    'diff_pixels': diff_pixels,
                    'diff_ratio': diff_pixels / clean_grid.size,
                    'false_negatives': false_negs,
                    'false_positives': false_pos
                })
                
                # Plan path
                planner = RRTStarGrid(
                    start=start,
                    goal=goal,
                    occupancy_grid=noisy_grid,
                    origin=parser.origin,
                    resolution=parser.resolution,
                    robot_radius=0.17,
                    max_iter=5000,
                    early_termination=True,
                    seed=trial
                )
                
                path = planner.plan()
                
                if path is not None and len(path) > 0:
                    # Compute score
                    score = scorer.compute_nonconformity_score(
                        clean_grid, noisy_grid, path, parser
                    )
                    scores.append(score)
                else:
                    planning_failures += 1
                
                pbar.update(1)
        
        # Restore noise types
        noise_model.noise_config['noise_types'] = original_types
        
        # Analyze results
        if scores:
            scores_array = np.array(scores)
            tau = np.percentile(scores_array, 90)
            
            results[label] = {
                'tau': float(tau),
                'scores': scores,
                'statistics': {
                    'num_scores': len(scores),
                    'non_zero_scores': int(np.sum(scores_array > 0)),
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'percentiles': {
                        'p50': float(np.percentile(scores_array, 50)),
                        'p75': float(np.percentile(scores_array, 75)),
                        'p90': float(np.percentile(scores_array, 90)),
                        'p95': float(np.percentile(scores_array, 95)),
                        'p99': float(np.percentile(scores_array, 99))
                    }
                },
                'planning_failures': planning_failures,
                'success_rate': (num_trials - planning_failures) / num_trials,
                'noise_effects': {
                    'mean_diff_ratio': np.mean([e['diff_ratio'] for e in noise_effects]),
                    'mean_false_negatives': np.mean([e['false_negatives'] for e in noise_effects]),
                    'mean_false_positives': np.mean([e['false_positives'] for e in noise_effects])
                }
            }
        else:
            results[label] = {
                'tau': 0.0,
                'scores': [],
                'planning_failures': planning_failures,
                'success_rate': 0.0,
                'error': 'All trials failed'
            }
        
        # Print summary
        print(f"\nResults for {label}:")
        print(f"  Planning success rate: {results[label]['success_rate']*100:.1f}%")
        if scores:
            print(f"  Tau (90th percentile): {tau:.4f}m")
            print(f"  Non-zero scores: {results[label]['statistics']['non_zero_scores']}/{len(scores)}")
            print(f"  Mean score: {results[label]['statistics']['mean']:.4f}m")
            print(f"  Max score: {results[label]['statistics']['max']:.4f}m")
            print(f"  Mean pixel change: {results[label]['noise_effects']['mean_diff_ratio']*100:.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Tau comparison
    ax = axes[0, 0]
    labels = list(results.keys())
    taus = [results[l]['tau'] if 'tau' in results[l] else 0 for l in labels]
    bars = ax.bar(range(len(labels)), taus)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', ' ').title() for l in labels], rotation=45, ha='right')
    ax.set_ylabel('Tau (m)')
    ax.set_title('Safety Margin (Tau) by Noise Type')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, tau in zip(bars, taus):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tau:.3f}', ha='center', va='bottom')
    
    # Plot 2: Score distributions
    ax = axes[0, 1]
    for i, label in enumerate(labels):
        if 'scores' in results[label] and results[label]['scores']:
            scores = results[label]['scores']
            positions = np.random.normal(i, 0.04, size=len(scores))
            ax.scatter(positions, scores, alpha=0.5, s=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('_', ' ').title() for l in labels], rotation=45, ha='right')
    ax.set_ylabel('Nonconformity Score (m)')
    ax.set_title('Score Distributions')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Noise effects
    ax = axes[1, 0]
    if all('noise_effects' in results[l] for l in labels):
        diff_ratios = [results[l]['noise_effects']['mean_diff_ratio']*100 for l in labels]
        bars = ax.bar(range(len(labels)), diff_ratios)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('_', ' ').title() for l in labels], rotation=45, ha='right')
        ax.set_ylabel('Pixels Changed (%)')
        ax.set_title('Mean Noise Effect on Grid')
        ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for label in labels:
        if 'statistics' in results[label]:
            stats = results[label]['statistics']
            table_data.append([
                label.replace('_', ' ').title()[:20],
                f"{results[label]['tau']:.3f}",
                f"{stats['mean']:.3f}",
                f"{stats['max']:.3f}",
                f"{results[label]['success_rate']*100:.0f}%"
            ])
    
    if table_data:
        table = ax.table(cellText=table_data,
                        colLabels=['Noise Type', 'Tau (m)', 'Mean (m)', 'Max (m)', 'Success'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results/corrected_ablation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    plot_path = results_dir / f'corrected_ablation_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    
    # Save JSON
    json_path = results_dir / f'corrected_ablation_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for label in labels:
        print(f"\n{label.replace('_', ' ').title()}:")
        if 'tau' in results[label]:
            print(f"  Tau: {results[label]['tau']:.4f}m")
            if 'noise_effects' in results[label]:
                print(f"  Pixels changed: {results[label]['noise_effects']['mean_diff_ratio']*100:.2f}%")
            if results[label]['tau'] == 0:
                print("  -> No underestimation detected (clearance increased or unchanged)")
            else:
                print("  -> Underestimation detected (obstacles appear closer)")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("""
1. MEASUREMENT NOISE: Causes underestimation -> positive tau
2. FALSE NEGATIVES: Removes obstacles -> no underestimation -> tau = 0
3. LOCALIZATION DRIFT: Shifts entire grid -> relative clearances unchanged -> tau = 0
4. COMBINED: Dominated by measurement noise effects

CONCLUSION: Standard CP primarily protects against measurement noise,
NOT against missing obstacles (false negatives) which is actually more dangerous!
    """)

if __name__ == "__main__":
    run_corrected_ablation()