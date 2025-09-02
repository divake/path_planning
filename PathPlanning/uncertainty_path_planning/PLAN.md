# Uncertainty-Aware Path Planning with Conformal Prediction
## Project Plan and Direction

### 🎯 Core Objective
Integrate Conformal Prediction (CP) methods with existing path planning algorithms to handle perception uncertainty, demonstrating safety guarantees with visual safety corridors.

---

## 📊 Three-Method Comparison Framework

### 1. **Naive Method**
- Uses planning algorithm directly on perceived environment
- No uncertainty handling
- Baseline for comparison
- Expected high collision rate with noisy perception

### 2. **Standard CP** (Current Focus)
- Fixed safety margin (τ) from calibration
- Provides 90% coverage guarantee
- Conservative but safe approach
- Uniform safety corridor visualization

### 3. **Learnable CP** (Future Work)
- Adaptive safety margins based on context
- Machine learning model predicts local uncertainty
- Efficient while maintaining safety
- Variable-width safety corridor

---

## 🎨 Key Innovation: Safety Corridor Visualization

### Concept (Inspired by Visual Odometry Bounds)
- **Center line**: Planned path (green)
- **Safety corridor**: Bounded region around path
  - Width represents uncertainty/safety margin
  - Color indicates risk level (blue=safe, red=high-risk)
- **Adaptive behavior**: 
  - Narrow in open spaces (low uncertainty)
  - Wide near obstacles (high uncertainty)
  - Widest at bottlenecks

### Visual Elements
```
Naive:          No corridor (just path)
Standard CP:    Uniform width corridor (τ = constant)
Learnable CP:   Adaptive width corridor (τ = f(context))
```

---

## 🔧 Implementation Strategy

### Phase 1: Naive vs Standard CP (Current)
1. **Environment Setup**
   - Use existing PathPlanning environments (51x31 grid)
   - Start with Search_2D/env.py obstacle configuration
   - A* algorithm as test planner

2. **Noise Model**
   - Gaussian noise on obstacle positions
   - σ = 0.5 grid cells (standard deviation)
   - Distance-dependent: `σ = 0.2 + 0.1 * distance_to_obstacle`
   - 5% miss rate, 3% false positive rate

3. **Calibration Process**
   ```
   For 1000 calibration runs:
   - Generate true obstacles
   - Add noise → perceived obstacles
   - Plan path on perceived
   - Execute on true environment
   - Record nonconformity score
   
   τ = 90th percentile of scores
   ```

4. **Standard CP Implementation**
   - Inflate obstacles by τ
   - Plan with inflated obstacles
   - Visualize with uniform corridor

5. **Monte Carlo Testing**
   - 10,000 runs for statistical significance
   - Measure: collision rate, path length, success rate
   - Verify 90% coverage guarantee

### Phase 2: Learnable CP (After Phase 1 Success)
- Simple MLP model (10 features → τ prediction)
- Features: local density, clearance, passage width
- Train on calibration data
- Show adaptive corridor visualization

---

## 📈 Metrics and Evaluation

### Safety Metrics
- **Collision rate**: Target < 10% for 90% coverage
- **Near-miss rate**: Within 0.5 units of collision
- **Safety margin distribution**: Statistics along path

### Efficiency Metrics
- **Path length ratio**: CP_path / optimal_path
- **Corridor area**: Total area covered (efficiency indicator)
- **Computation overhead**: Time comparison

### Robustness Metrics
- **Noise sensitivity**: Performance vs noise level
- **Environment generalization**: Across different maps
- **Failure analysis**: Where/why methods fail

---

## 🗂️ Directory Structure
```
uncertainty_path_planning/
├── PLAN.md (this file)
├── src/
│   ├── noise_models.py       # Perception uncertainty
│   ├── standard_cp.py        # Standard CP implementation
│   ├── learnable_cp.py       # Learnable CP (future)
│   ├── evaluation.py         # Metrics and testing
│   └── visualization.py      # Safety corridor viz
├── experiments/
│   ├── calibration/          # Calibration data
│   ├── results/              # Test results
│   └── figures/              # Visualizations
└── configs/
    └── experiment_config.yaml # Parameters
```

---

## 🔬 Experimental Design

### Environment Types (Future)
1. **Original**: PathPlanning default environment
2. **Narrow Corridors**: Test conservative margins
3. **Open Space**: Test efficiency
4. **Cluttered**: High obstacle density
5. **Mixed**: Varying complexity

### Parameter Ranges
- Noise levels: σ ∈ [0.3, 0.5, 0.7]
- Coverage targets: α ∈ [0.1, 0.05, 0.01]
- Calibration sizes: [500, 1000, 2000]

---

## 📝 Key Design Decisions

### 1. **Nonconformity Score**
- Maximum perception error along planned path
- Distance between perceived and true obstacle positions
- Captures worst-case uncertainty

### 2. **Safety Margin Application**
- **Option A**: Inflate obstacles (current plan)
- **Option B**: Modify cost function
- **Option C**: Post-process path

### 3. **Visualization Strategy**
- Perpendicular bounds at each path point
- Smooth corridor using spline interpolation
- Color gradient based on local risk
- Transparency for confidence levels

---

## 🚀 Success Criteria

### Phase 1 (Standard CP)
- ✅ Collision rate reduction > 80% vs Naive
- ✅ Maintains 90% coverage guarantee
- ✅ Clear corridor visualization
- ✅ Reasonable path length increase (< 30%)

### Phase 2 (Learnable CP)
- ✅ Path length improvement > 20% vs Standard CP
- ✅ Maintains safety guarantees
- ✅ Adaptive corridor visualization
- ✅ Generalizes across environments

---

## 📅 Timeline

### Week 1
- [x] Repository setup and analysis
- [ ] Implement noise model
- [ ] Standard CP calibration
- [ ] Basic corridor visualization

### Week 2
- [ ] Monte Carlo experiments
- [ ] Results analysis
- [ ] Learnable CP implementation
- [ ] Comparative evaluation

### Week 3
- [ ] Extended experiments
- [ ] Paper writing
- [ ] Final visualizations

---

## 🔍 Critical Questions to Resolve

1. **Corridor computation at sharp turns?**
   - Union of perpendicular bounds
   - Clip to feasible space
   
2. **How to handle dynamic τ for visualization?**
   - Even Standard CP could show distance-based width
   - Pure CP vs enhanced visualization

3. **Statistical validation approach?**
   - Bootstrap confidence intervals
   - Paired tests for comparison

---

## 💡 Key Insights to Emphasize

1. **Perception-Planning Gap**: First principled approach using CP
2. **Visual Interpretability**: Safety corridors show guarantees
3. **Practical Impact**: Works with any planner (A*, RRT*, etc.)
4. **Theoretical Soundness**: Maintains formal coverage
5. **Computational Efficiency**: Minimal overhead

---

## 📚 References and Inspiration

- Visual Odometry bounds visualization (previous work)
- Conformal Prediction theory (Vovk et al.)
- PathPlanning repository structure
- ICRA paper requirements

---

## 🎯 Next Immediate Steps

1. Create noise model implementation
2. Set up calibration framework
3. Implement Standard CP with A*
4. Create safety corridor visualization
5. Run initial experiments (100 runs)
6. Validate coverage guarantee

---

## 📝 Notes

- Keep implementation simple and modular
- Follow PathPlanning code style
- Prioritize clear visualization over complex math
- Document coverage guarantees clearly
- Test incrementally, don't write everything at once

---

*Last Updated: [Current Date]*
*This document guides the implementation and ensures consistency with discussed goals.*