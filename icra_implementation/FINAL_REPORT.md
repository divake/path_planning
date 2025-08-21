# 
ICRA EXPERIMENT FINAL REPORT

Generated: 20250821T03:32:29.434321
Total Scenarios: 100

KEY FINDINGS:

1. SAFETY IMPROVEMENT:
    Learnable CP reduces collisions by 88.2% vs Naive
    Learnable CP reduces collisions by 62.1% vs Ensemble

2. EFFICIENCY:
    Learnable CP paths are only 6.8% longer than Naive
    Learnable CP paths are 6.3% shorter than Ensemble

3. COVERAGE GUARANTEE:
    Achieved 0.949 coverage (target: 0.950)
    Coverage std: 0.0175

4. ADAPTIVITY:
    Average adaptivity score: 0.695
    Shows strong adaptation to environment complexity


ENVIRONMENTSPECIFIC PERFORMANCE:


SPARSE:
  naive          : Collision0.119, Success96.0%
  ensemble       : Collision0.033, Success100.0%
  learnable_cp   : Collision0.019, Success100.0%

MODERATE:
  naive          : Collision0.255, Success100.0%
  ensemble       : Collision0.069, Success100.0%
  learnable_cp   : Collision0.036, Success100.0%

DENSE:
  naive          : Collision0.325, Success84.0%
  ensemble       : Collision0.106, Success100.0%
  learnable_cp   : Collision0.038, Success100.0%

NARROW PASSAGE:
  naive          : Collision0.525, Success48.0%
  ensemble       : Collision0.173, Success100.0%
  learnable_cp   : Collision0.052, Success100.0%


STATISTICAL ANALYSIS:

Naive vs Learnable CP collision rates:
  tstatistic: 13.767
  pvalue: 0.000000
  Significant: Yes


CONCLUSIONS:

1. Learnable CP provides superior safety with minimal performance cost
2. Adaptive uncertainty quantification works effectively across environments
3. Coverage guarantees are maintained near target levels
4. Method shows strong potential for realworld deployment


END OF REPORT
