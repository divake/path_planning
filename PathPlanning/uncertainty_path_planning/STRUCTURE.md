# Directory Structure - Uncertainty Path Planning

## ✅ Clean Organization

```
uncertainty_path_planning/
│
├── 📁 discrete_planner/          # Grid-based A* with discrete CP
│   ├── discrete_cp_system.py     # Main discrete CP implementation
│   ├── step1_baseline_test.py    # A* baseline
│   ├── step2_noise_model.py      # Thickness noise (failed approach)
│   ├── step3_verify_problem.py   # Path analysis
│   ├── step4_collision_analysis.py # Finding collision-inducing noise
│   ├── step5_naive_with_collisions.py # Wall thinning (36.7% collisions)
│   ├── step6_standard_cp.py      # Standard CP (0.06% collisions)
│   ├── final_comparison.py       # 10,000 trial Monte Carlo
│   ├── explain_*.py              # 3 explanation files
│   ├── README.md                 # Documentation
│   └── 📁 results/               # 11 visualizations
│       ├── step1-6_*.png        # Step-by-step results
│       └── *_explained.png       # Explanations
│
├── 📁 continuous_planner/        # RRT* with continuous CP
│   ├── continuous_cp_system.py   # Main continuous CP implementation
│   ├── README.md                 # Documentation
│   └── 📁 results/               # Continuous visualizations
│       └── continuous_system.png
│
├── 📁 results/                   # Comparison results
│   ├── discrete_vs_continuous_comparison.png
│   └── inflation_comparison.png
│
├── compare_discrete_continuous.py # Main comparison script
├── PLAN.md                       # Original project plan
├── README.md                     # Main documentation
├── STRUCTURE.md                 # This file
└── Visual_odometry.jpg          # Reference image
```

## Key Features

### Discrete Planner
- **Algorithm**: A* on 51×31 grid
- **τ values**: 0, 1, or 2 (integers only)
- **Coverage**: Discrete jumps (~85% or ~99%)
- **Files run from**: Main uncertainty_path_planning/ directory

### Continuous Planner  
- **Algorithm**: RRT* in continuous space
- **τ values**: Any float (0.12, 0.15, etc.)
- **Coverage**: Any percentage precisely
- **Files run from**: Main uncertainty_path_planning/ directory

## Running Code

All scripts should be run from the main `uncertainty_path_planning/` directory:

```bash
# From uncertainty_path_planning/ directory:

# Run discrete system
python discrete_planner/discrete_cp_system.py
python discrete_planner/step1_baseline_test.py  # etc.

# Run continuous system
python continuous_planner/continuous_cp_system.py

# Run comparison
python compare_discrete_continuous.py
```

## Path Configuration

All files are configured to:
- Save results to their respective `*/results/` directories
- Import modules using relative paths from main directory
- Work correctly when run from `uncertainty_path_planning/`

## Clean & Expandable

This structure makes it easy to:
- Add new planners (just create new directory)
- Compare methods (use comparison script as template)
- Track development (step files show progression)
- Understand concepts (explanation files included)