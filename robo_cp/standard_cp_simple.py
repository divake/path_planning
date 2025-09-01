"""
Simple Standard CP Calibration Demo
Shows the key concept: tau = (1-alpha) quantile of perception errors
"""

import numpy as np
import matplotlib.pyplot as plt

def simple_calibration_demo():
    """
    Demonstrate Standard CP calibration process.
    """
    print("="*60)
    print("STANDARD CP: SIMPLE CALIBRATION DEMO")
    print("="*60)
    
    # Simulate perception errors (residuals) from calibration data
    # In reality, these come from your dataset
    np.random.seed(42)
    
    # Mix of different noise levels
    noise_levels = [0.1, 0.3, 0.5]
    residuals = []
    
    print("\n1. Simulating perception errors from calibration data...")
    for _ in range(1000):
        noise = np.random.choice(noise_levels)
        # Perception error follows roughly noise * chi distribution
        error = np.abs(np.random.normal(0, noise)) + np.abs(np.random.normal(0, noise))
        residuals.append(error)
    
    residuals = np.array(residuals)
    
    print(f"   Generated {len(residuals)} perception errors")
    print(f"   Mean error: {np.mean(residuals):.3f}")
    print(f"   Std deviation: {np.std(residuals):.3f}")
    
    # Calculate tau as (1-alpha) quantile
    alpha = 0.1  # For 90% coverage
    tau = np.quantile(residuals, 1 - alpha)
    
    print(f"\n2. Calculating tau for {(1-alpha)*100:.0f}% coverage:")
    print(f"   50th percentile: {np.quantile(residuals, 0.50):.3f}")
    print(f"   75th percentile: {np.quantile(residuals, 0.75):.3f}")
    print(f"   85th percentile: {np.quantile(residuals, 0.85):.3f}")
    print(f"   90th percentile: {np.quantile(residuals, 0.90):.3f} ← This is tau!")
    print(f"   95th percentile: {np.quantile(residuals, 0.95):.3f}")
    
    print(f"\n" + "="*40)
    print(f"CALIBRATED TAU = {tau:.3f}")
    print(f"="*40)
    
    # Test coverage
    print(f"\n3. Testing coverage guarantee:")
    test_errors = []
    for _ in range(500):
        noise = np.random.choice(noise_levels)
        error = np.abs(np.random.normal(0, noise)) + np.abs(np.random.normal(0, noise))
        test_errors.append(error)
    
    test_errors = np.array(test_errors)
    coverage = np.mean(test_errors <= tau)
    
    print(f"   Test set coverage: {coverage*100:.1f}%")
    if coverage >= 0.9:
        print(f"   ✓ Coverage guarantee satisfied!")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, 
               label=f'τ = {tau:.3f} (90th percentile)')
    ax.set_xlabel('Perception Error')
    ax.set_ylabel('Density')
    ax.set_title('Calibration: Distribution of Perception Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CDF
    ax = axes[1]
    sorted_residuals = np.sort(residuals)
    cumulative = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)
    ax.plot(sorted_residuals, cumulative, 'b-', linewidth=2)
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ = {tau:.3f}')
    ax.axhline(0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='90% coverage')
    ax.set_xlabel('Perception Error')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Empirical CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Standard CP: τ is the 90th Percentile of Calibration Errors', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/standard_cp_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return tau

def show_tau_for_different_scenes():
    """
    Show that different environments need different tau values.
    """
    print("\n" + "="*60)
    print("DIFFERENT SCENES NEED DIFFERENT TAU")
    print("="*60)
    
    scenarios = [
        ("Indoor office", 0.05, "Low noise, structured environment"),
        ("Outdoor park", 0.2, "Moderate noise, natural obstacles"),
        ("Construction site", 0.5, "High noise, dynamic obstacles"),
        ("Foggy conditions", 0.8, "Very high uncertainty")
    ]
    
    taus = []
    for name, noise, description in scenarios:
        # Simulate errors for this scenario
        errors = np.abs(np.random.normal(0, noise, 1000)) + \
                 np.abs(np.random.normal(0, noise, 1000))
        tau = np.quantile(errors, 0.9)
        taus.append(tau)
        print(f"\n{name}:")
        print(f"  {description}")
        print(f"  Typical noise level: σ={noise:.2f}")
        print(f"  Calibrated τ = {tau:.3f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    names = [s[0] for s in scenarios]
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = plt.bar(names, taus, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Calibrated τ (Safety Margin)', fontsize=12)
    plt.title('Different Environments Require Different Safety Margins', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, tau in zip(bars, taus):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'τ={tau:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/tau_different_scenes.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main demo."""
    # Basic calibration
    tau = simple_calibration_demo()
    
    # Show different scenes
    show_tau_for_different_scenes()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("1. Standard CP uses the (1-α) quantile of calibration errors as τ")
    print("2. This τ guarantees (1-α)% coverage on new test data")
    print("3. Different environments need different τ values")
    print("4. τ is fixed for a given environment after calibration")
    print(f"\nFor our wall environment with mixed noise: τ ≈ {tau:.3f}")

if __name__ == "__main__":
    main()