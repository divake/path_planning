"""
Analyze the range of tau values
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_tau_range():
    """
    Show the theoretical and practical range of tau values.
    """
    print("="*60)
    print("TAU VALUE RANGE ANALYSIS")
    print("="*60)
    
    # Theoretical range
    print("\n1. THEORETICAL RANGE:")
    print("   τ ∈ [0, ∞)")
    print("   - τ = 0: No safety margin (accept all risk)")
    print("   - τ → ∞: Infinite safety margin (extremely conservative)")
    
    # Practical range for robotics
    print("\n2. PRACTICAL RANGE IN ROBOTICS:")
    print("   Typically τ ∈ [0.1, 5.0] grid units")
    print("   - τ < 0.5: Low safety margin (high precision sensors)")
    print("   - τ ∈ [0.5, 2.0]: Normal range for most scenarios")
    print("   - τ > 2.0: High safety margin (poor sensors/visibility)")
    
    # Show how tau changes with coverage level
    print("\n3. TAU VS COVERAGE LEVEL (for σ=0.3 noise):")
    
    # Simulate errors with σ=0.3
    np.random.seed(42)
    errors = []
    for _ in range(10000):
        error = np.abs(np.random.normal(0, 0.3)) + np.abs(np.random.normal(0, 0.3))
        errors.append(error)
    errors = np.array(errors)
    
    coverage_levels = [50, 70, 80, 85, 90, 95, 99, 99.9]
    taus = []
    
    for coverage in coverage_levels:
        tau = np.quantile(errors, coverage/100)
        taus.append(tau)
        print(f"   {coverage:5.1f}% coverage → τ = {tau:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Tau vs Coverage
    ax = axes[0]
    ax.plot(coverage_levels, taus, 'b-o', linewidth=2, markersize=8)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='τ=1.0 (our calibrated value)')
    ax.set_xlabel('Coverage Level (%)')
    ax.set_ylabel('Required τ')
    ax.set_title('How τ Increases with Coverage Requirements')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Highlight 90% point
    idx_90 = coverage_levels.index(90)
    ax.plot(90, taus[idx_90], 'ro', markersize=12)
    ax.annotate(f'90% → τ={taus[idx_90]:.3f}', 
                xy=(90, taus[idx_90]), xytext=(85, taus[idx_90]+0.2),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Distribution with different tau thresholds
    ax = axes[1]
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    ax.plot(sorted_errors, cumulative * 100, 'b-', linewidth=2, label='Error CDF')
    
    # Mark different tau values
    tau_examples = [0.5, 1.0, 1.5, 2.0]
    colors = ['green', 'orange', 'red', 'darkred']
    for tau, color in zip(tau_examples, colors):
        coverage = np.mean(errors <= tau) * 100
        ax.axvline(tau, color=color, linestyle='--', alpha=0.7)
        ax.text(tau + 0.05, 50, f'τ={tau}\n({coverage:.1f}%)', 
                fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel('Perception Error')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Achieved by Different τ Values')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])
    
    plt.suptitle('Tau Value Range Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tau_range_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n4. FACTORS AFFECTING TAU RANGE:")
    print("   - Sensor noise level (σ)")
    print("   - Environment complexity")
    print("   - Safety requirements (α)")
    print("   - Robot size and dynamics")
    
    print("\n5. FOR OUR ENVIRONMENT:")
    print("   - Mixed noise σ ∈ [0.1, 0.3, 0.5]")
    print("   - 90% coverage requirement (α=0.1)")
    print("   - Calibrated τ = 1.001")
    print("   - This is in the normal range [0.5, 2.0]")

if __name__ == "__main__":
    analyze_tau_range()