# Discrete Planner with Conformal Prediction

This directory contains all code related to **discrete (grid-based) path planning** with conformal prediction using A* algorithm.

## Main Files

### Core Implementation
- `discrete_cp_system.py` - Complete discrete CP implementation with integer τ values

### Step-by-Step Development
- `step1_baseline_test.py` - Initial A* baseline verification
- `step2_noise_model.py` - Thickness-based noise model (initial attempt)
- `step3_verify_problem.py` - Path analysis with noise
- `step4_collision_analysis.py` - Finding collision-inducing noise models
- `step5_naive_with_collisions.py` - Naive method with wall thinning (36.7% collision rate)
- `step6_standard_cp.py` - Standard CP implementation (0.06% collision rate)

### Analysis & Explanation
- `explain_collision_mechanism.py` - How wall thinning causes collisions
- `explain_tau_calculation.py` - How τ is calculated from calibration data
- `explain_calibration_dataset.py` - Understanding calibration without path planning
- `final_comparison.py` - Comprehensive 10,000 trial Monte Carlo evaluation

## Key Results

- **Noise Model**: 5% wall thinning
- **τ Value**: 1 or 2 (integer cells only)
- **Collision Reduction**: 36.7% → 0.06%
- **Path Length Increase**: ~14%
- **Coverage**: Discrete jumps (~85% or ~99%)

## Results Directory

All visualizations and plots are saved in `results/` including:
- Step-by-step visualizations (step1-6)
- Explanation diagrams
- Final comparison plots