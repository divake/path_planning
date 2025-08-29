# Uncertainty Methods Comparison for Path Planning

Simple implementation comparing three uncertainty quantification methods for robot path planning using the python_motion_planning library.

## Methods

1. **Naive**: No obstacle inflation (baseline)
2. **Traditional CP**: Fixed margin inflation around all obstacles
3. **Learnable CP**: Adaptive inflation (to be implemented with ML)

## Files

- `test_methods.py` - Main test script comparing all three methods
- `results/methods_comparison.png` - Visualization of the three methods
- `README.md` - This file

## Usage

```bash
python test_methods.py
```

This will:
1. Create a simple wall environment (same as python_motion_planning examples)
2. Run A* path planning with each uncertainty method
3. Generate comparison visualization
4. Run Monte Carlo analysis (100 trials with simulated noise)

## Results

From the test run:
- **Naive**: Shortest path (cost 54.0) but 29% collision rate with noise
- **Traditional CP**: Longer path (cost 56.3) but only 5% collision rate  
- **Learnable CP**: Balanced approach (cost 55.5) with 4% collision rate

## Visualization

The generated image shows:
- Gray blocks: Original wall obstacles
- Yellow blocks: Safety margins (for Traditional CP and Learnable CP)
- Blue path: Planned trajectory
- Green circle: Start position
- Red star: Goal position

## Dependencies

- python_motion_planning (included as submodule)
- numpy
- matplotlib

## Next Steps

- Implement proper ML-based adaptive margins for Learnable CP
- Add more complex environments
- Test with real robot data