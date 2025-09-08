#!/usr/bin/env python3
"""
Test run with fewer trials to verify everything works
"""

import sys
sys.path.append('/mnt/ssd1/divake/path_planning/PathPlanning/uncertainty_path_planning/continuous_planner')

from run_icra_complete_experiments import ICRAExperimentRunner

# Create experiment runner with fewer trials for testing
runner = ICRAExperimentRunner(
    num_trials=50,  # Small number for quick test
    noise_level=0.15   # 15% perception noise
)

print("Running quick test with 50 trials...")

# Run experiments on just a few environments
runner.training_envs = {'passages': runner.training_envs['passages']}
runner.test_envs = {'zigzag': runner.test_envs['zigzag']}
runner.all_envs = {**runner.training_envs, **runner.test_envs}

# Run all experiments
results = runner.run_all_experiments()

# Generate tables
runner.generate_detailed_tables()

# Generate plots
runner.generate_publication_plots()

# Print summary
runner.print_final_summary()

print("\n✓ Test run complete! If successful, run the full experiment.")