#!/usr/bin/env python3
"""
Quick test with fewer trials to verify everything works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.run_experiments import ExperimentRunner

if __name__ == "__main__":
    print("Running quick test with 10 trials...")
    
    # Run with fewer trials for quick test
    runner = ExperimentRunner(
        env_size=(30, 30),  # Smaller environment
        num_obstacles=10,    # Fewer obstacles
        num_trials=10,       # Only 10 trials
        noise_sigma=0.3      # Less noise
    )
    
    runner.run_full_experiment()
    
    print("\nQuick test complete! Check the results/ folder for outputs.")
    print("To run full experiment with 100 trials, use:")
    print("  python scripts/run_experiments.py")