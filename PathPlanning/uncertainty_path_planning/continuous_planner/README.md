# Continuous Planner with Conformal Prediction

This directory contains all code related to **continuous path planning** with conformal prediction using RRT* algorithm.

## Main Files

### Core Implementation
- `continuous_cp_system.py` - Complete continuous CP implementation with RRT* planner

## Key Features

- **Planner**: RRT* (Rapidly-exploring Random Tree Star)
- **Space**: Continuous 2D (50.0 × 30.0 units)
- **Noise Model**: Continuous obstacle shrinking/expansion
- **Nonconformity Score**: Continuous penetration depth (0.00 to 0.15)

## Key Results

- **τ Value**: 0.12 (continuous, precise value)
- **Flexibility**: Can achieve any τ (0.08, 0.10, 0.12, 0.15, etc.)
- **Coverage Control**: Smooth, can achieve any desired percentage
- **Collision Rate**: 6.5% → 0.5% with CP
- **Path Length**: 47.9 → 52.6 units

## Advantages Over Discrete

1. **Fine-grained τ**: Any value possible (not just integers)
2. **Precise Coverage**: Can achieve exactly 90%, 95%, or any target
3. **Smooth Inflation**: Obstacles inflate by exact amounts
4. **Better Trade-offs**: Can fine-tune safety vs efficiency

## Results Directory

Visualizations saved in `results/`:
- `continuous_system.png` - Example of continuous CP in action