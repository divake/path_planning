# Robo-CP: Uncertainty-Aware Path Planning

This repository implements and compares three uncertainty quantification methods for robot path planning:

1. **Naive Method**: No uncertainty consideration (baseline)
2. **Traditional Conformal Prediction**: Fixed uniform safety margins
3. **Learnable Conformal Prediction**: Adaptive safety margins based on learned features

## Directory Structure

```
robo_cp/
├── src/                      # Core source code
│   ├── feature_extraction.py # Extract features from environment
│   └── planner_wrapper.py    # Wrapper for path planners
├── environments/             # Environment implementations
│   └── base_environment.py   # Environment with uncertainty handling
├── methods/                  # Uncertainty methods
│   ├── naive_method.py       # Baseline (no uncertainty)
│   ├── traditional_cp_method.py  # Fixed margins
│   └── learnable_cp_method.py    # Adaptive margins with MLP
├── scripts/                  # Experiment scripts
│   ├── run_experiments.py    # Main experiment runner
│   └── test_setup.py         # Test installation
├── results/                  # Output directory
├── data/                     # Data storage
└── visualization/            # Plotting utilities
```

## Installation

1. Clone the python_motion_planning repository (already done):
```bash
git clone https://github.com/ai-winter/python_motion_planning.git
```

2. Install dependencies:
```bash
pip install numpy matplotlib scipy torch pandas
```

3. Test the setup:
```bash
cd robo_cp
python scripts/test_setup.py
```

## Usage

### Quick Test (10 trials)
```bash
python run_quick_test.py
```

### Full Experiment (100 trials)
```bash
python scripts/run_experiments.py
```

### Custom Experiment
```python
from scripts.run_experiments import ExperimentRunner

runner = ExperimentRunner(
    env_size=(50, 50),      # Environment size
    num_obstacles=20,        # Number of obstacles
    num_trials=100,          # Monte Carlo trials
    noise_sigma=0.5         # Noise level
)

runner.run_full_experiment()
```

## Key Features

### Environment
- Random obstacle generation with clusters and narrow passages
- Noise injection for Monte Carlo testing
- Configurable obstacle inflation based on uncertainty method

### Feature Extraction
- Distance to nearest obstacle
- Local obstacle density
- Passage width estimation
- Boundary distances
- Quadrant-based obstacle distribution

### Methods

**Naive (Baseline)**
- No obstacle inflation
- Direct path planning on original obstacles

**Traditional CP**
- Fixed margin inflation (default: 1.5 units)
- Uniform safety buffer around all obstacles
- Configurable confidence level

**Learnable CP**
- Neural network predicts uncertainty scores
- Adaptive margins based on local features
- Range: 0.1 to 2.0 units
- Trainable and calibratable

### Metrics Tracked
- **Safety**: Collision rate, minimum clearance
- **Efficiency**: Path length, computation time
- **Success**: Goal reached rate

## Results

The experiment outputs:
- `summary.csv`: Quantitative comparison of methods
- `method_comparison.png`: Visual comparison
- `detailed_results.pkl`: Full trial data

## Expected Outcomes

With proper training, Learnable CP should show:
- Lower collision rates than Naive
- Shorter paths than Traditional CP
- Better safety-efficiency tradeoff

## Extensions

The framework is designed to be easily extensible:

1. **Add new planners**: Modify `planner_wrapper.py`
2. **Add new uncertainty methods**: Create new file in `methods/`
3. **Add new environments**: Extend `base_environment.py`
4. **Add new features**: Modify `feature_extraction.py`

## Integration with Existing Code

This implementation can use the trained models from the ICRA implementation:
```python
learnable_cp = LearnableCPMethod(
    model_path="../icra_implementation/checkpoints/cp_model.pth"
)
```

## Next Steps

1. Train the learnable CP model on diverse scenarios
2. Test on standard benchmarks (BARN, MovingAI)
3. Deploy on Scout Mini robot
4. Compare with more baseline methods

## Citation

If you use this code, please cite:
```
@inproceedings{robocp2025,
  title={Learnable Conformal Prediction for Uncertainty-Aware Path Planning},
  author={Your Name},
  booktitle={ICRA},
  year={2025}
}
```