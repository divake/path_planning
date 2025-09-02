# Uncertainty-Aware Path Planning with Conformal Prediction

This project implements and compares uncertainty-aware path planning using Conformal Prediction (CP) for both discrete and continuous planners.

## Project Structure

```
uncertainty_path_planning/
├── discrete_planner/       # Grid-based A* with discrete CP
│   ├── *.py               # Implementation files
│   └── results/           # Visualizations and plots
├── continuous_planner/     # RRT* with continuous CP
│   ├── *.py               # Implementation files
│   └── results/           # Visualizations and plots
├── compare_discrete_continuous.py  # Comparison analysis
├── results/               # Comparison results
└── PLAN.md               # Original project plan
```

## Key Concepts

### Conformal Prediction (CP)
A statistical framework that provides probabilistic safety guarantees for path planning under perception uncertainty.

### The Process
1. **Calibration**: Measure perception errors (no path planning needed!)
2. **Compute τ**: Safety margin as quantile of error distribution
3. **Inflate Obstacles**: Expand obstacles by τ for conservative planning
4. **Guarantee**: (1-ε) probability of collision-free paths

## Main Results

### Discrete Planner (A*)
- **τ = 1 or 2** (integer cells)
- Collision rate: 36.7% → 0.06%
- Limited to discrete coverage levels

### Continuous Planner (RRT*)
- **τ = 0.12** (precise value)
- Collision rate: 6.5% → 0.5%
- Can achieve any coverage level

## Key Finding

**Discrete planners** can only have integer τ values (0, 1, 2), leading to coarse coverage control (~85% or ~99%).

**Continuous planners** allow any τ value (0.12, 0.15, 1.57), enabling precise coverage targets (exactly 90%, 95%, etc.).

## Running the Code

```bash
# Run discrete system
python discrete_planner/discrete_cp_system.py

# Run continuous system
python continuous_planner/continuous_cp_system.py

# Compare both systems
python compare_discrete_continuous.py
```

## Publications
This implements concepts from conformal prediction literature applied to robotic path planning.