# Uncertainty-Aware Path Planning with Conformal Prediction

Clean implementation for ICRA 2025 submission comparing Naive, Standard CP, and Learnable CP methods.

## Framework Structure

### Core Files (8 files only)
- `continuous_environment.py` - Environment definition with obstacles
- `continuous_standard_cp.py` - Standard Conformal Prediction implementation
- `learnable_cp_final.py` - Learnable CP with adaptive tau prediction
- `rrt_star_planner.py` - RRT* path planner with robot radius
- `cross_validation_environments.py` - Training and test environments
- `continuous_visualization.py` - Visualization utilities
- `proper_monte_carlo_evaluation.py` - Monte Carlo evaluation framework
- `framework_verification.py` - Framework verification tests

### Main Entry Point
- `run_experiments.py` - Run all experiments and generate paper figures

## Quick Start

### 1. Verify Framework
```bash
python framework_verification.py
```

### 2. Run Full Experiments
```bash
python run_experiments.py
```

## Results Location
- Experiment results: `results/experiment_results_*.json`
- Figures: `results/figures/`
