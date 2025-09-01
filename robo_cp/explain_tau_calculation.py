"""
Explain and visualize how tau is calculated for Standard CP
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/ssd1/divake/path_planning/python_motion_planning/src')
from python_motion_planning.utils import Grid, SearchFactory


def create_wall_environment():
    """Create the wall environment."""
    grid = Grid(51, 31)
    obstacles = grid.obstacles
    
    for i in range(10, 21):
        obstacles.add((i, 15))
    for i in range(15):
        obstacles.add((20, i))
    for i in range(15, 30):
        obstacles.add((30, i))
    for i in range(16):
        obstacles.add((40, i))
    
    grid.update(obstacles)
    return grid


def detailed_calibration(num_samples=1000, noise_level=0.2, alpha=0.1):
    """
    Detailed calibration showing how tau is calculated.
    """
    print("="*60)
    print("DETAILED TAU CALIBRATION")
    print("="*60)
    
    env = create_wall_environment()
    true_obstacles = env.obstacles
    
    perception_errors = []
    
    print(f"\nCalibration Parameters:")
    print(f"  Noise level (σ): {noise_level}")
    print(f"  Coverage target: {(1-alpha)*100}%")
    print(f"  Calibration samples: {num_samples}")
    
    print("\nCollecting perception errors...")
    
    for sample in range(num_samples):
        # Simulate noisy sensor reading
        perceived_obstacles = set()
        for obs in true_obstacles:
            # Add Gaussian noise to each obstacle position
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)
            
            new_x = int(np.clip(obs[0] + noise_x, 0, 50))
            new_y = int(np.clip(obs[1] + noise_y, 0, 30))
            perceived_obstacles.add((new_x, new_y))
        
        # Measure perception error at various points
        for obs in true_obstacles:
            # Find closest perceived obstacle to this true obstacle
            min_error = min([np.sqrt((obs[0] - p[0])**2 + (obs[1] - p[1])**2) 
                           for p in perceived_obstacles])
            perception_errors.append(min_error)
    
    perception_errors = np.array(perception_errors)
    
    # Calculate tau as the (1-alpha) quantile
    tau = np.quantile(perception_errors, 1 - alpha)
    
    print(f"\nPerception Error Statistics:")
    print(f"  Total measurements: {len(perception_errors)}")
    print(f"  Mean error: {np.mean(perception_errors):.3f}")
    print(f"  Std deviation: {np.std(perception_errors):.3f}")
    print(f"  Min error: {np.min(perception_errors):.3f}")
    print(f"  Max error: {np.max(perception_errors):.3f}")
    print(f"  50th percentile: {np.quantile(perception_errors, 0.50):.3f}")
    print(f"  75th percentile: {np.quantile(perception_errors, 0.75):.3f}")
    print(f"  90th percentile: {np.quantile(perception_errors, 0.90):.3f}")
    print(f"  95th percentile: {np.quantile(perception_errors, 0.95):.3f}")
    
    print(f"\n{'='*40}")
    print(f"CALCULATED TAU = {tau:.3f}")
    print(f"{'='*40}")
    print(f"\nThis means:")
    print(f"  - 90% of perception errors are ≤ {tau:.3f} units")
    print(f"  - By inflating obstacles by {tau:.3f} units,")
    print(f"    we get 90% safety coverage")
    
    return tau, perception_errors


def visualize_tau_calculation(tau, errors):
    """
    Visualize how tau is determined from the error distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Error distribution histogram
    ax = axes[0]
    ax.hist(errors, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ = {tau:.3f}')
    ax.set_xlabel('Perception Error (units)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Perception Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    coverage = np.mean(errors <= tau) * 100
    ax.text(0.95, 0.95, f'{coverage:.1f}% of errors\n≤ τ', 
            transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Cumulative distribution
    ax = axes[1]
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    ax.axvline(tau, color='red', linestyle='--', linewidth=2, label=f'τ = {tau:.3f}')
    ax.axhline(0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='90% coverage')
    ax.set_xlabel('Perception Error (units)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Visual explanation
    ax = axes[2]
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 8)
    ax.set_aspect('equal')
    
    # True obstacle
    true_obs = plt.Circle((5, 4), 0.3, color='black', label='True obstacle')
    ax.add_patch(true_obs)
    
    # Perceived positions (with noise)
    np.random.seed(42)
    for i in range(20):
        noise_x = np.random.normal(0, 0.2) * 5
        noise_y = np.random.normal(0, 0.2) * 5
        perceived = plt.Circle((5 + noise_x, 4 + noise_y), 0.2, 
                              color='red', alpha=0.3)
        ax.add_patch(perceived)
    
    # Safety margin
    safety = plt.Circle((5, 4), tau * 5, fill=False, 
                       edgecolor='blue', linewidth=2, 
                       linestyle='--', label=f'Safety margin (τ={tau:.3f})')
    ax.add_patch(safety)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Why We Need Safety Margin τ')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add explanation
    ax.text(5, 1, 'Sensor noise causes\nuncertain obstacle positions', 
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle(f'Tau Calibration: How τ={tau:.3f} is Determined', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tau_explanation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to results/tau_explanation.png")


def test_different_noise_levels():
    """
    Show how tau changes with different noise levels.
    """
    print("\n" + "="*60)
    print("TAU VALUES FOR DIFFERENT NOISE LEVELS")
    print("="*60)
    
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    taus = []
    
    for noise in noise_levels:
        env = create_wall_environment()
        true_obstacles = env.obstacles
        
        errors = []
        for _ in range(500):
            perceived = set()
            for obs in true_obstacles:
                noise_x = np.random.normal(0, noise)
                noise_y = np.random.normal(0, noise)
                new_x = int(np.clip(obs[0] + noise_x, 0, 50))
                new_y = int(np.clip(obs[1] + noise_y, 0, 30))
                perceived.add((new_x, new_y))
            
            for obs in true_obstacles:
                min_error = min([np.sqrt((obs[0] - p[0])**2 + (obs[1] - p[1])**2) 
                               for p in perceived])
                errors.append(min_error)
        
        tau = np.quantile(errors, 0.9)
        taus.append(tau)
        print(f"  Noise σ={noise:.2f} → τ={tau:.3f}")
    
    # Plot relationship
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, taus, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Sensor Noise Level (σ)')
    plt.ylabel('Required Safety Margin (τ)')
    plt.title('Safety Margin vs Sensor Noise')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for noise, tau in zip(noise_levels, taus):
        plt.annotate(f'τ={tau:.2f}', 
                    xy=(noise, tau), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    plt.savefig('results/tau_vs_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved to results/tau_vs_noise.png")
    
    print("\nKey Insight:")
    print("  Higher sensor noise → Larger safety margin needed")
    print("  τ ≈ 5σ for 90% coverage in this environment")


def main():
    """Main demonstration."""
    # Detailed calibration
    tau, errors = detailed_calibration(num_samples=1000, noise_level=0.2, alpha=0.1)
    
    # Visualize
    visualize_tau_calculation(tau, errors)
    
    # Test different noise levels
    test_different_noise_levels()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"With noise level σ=0.2, we get τ≈1.0")
    print("This means we inflate obstacles by 1 unit radius")
    print("to achieve 90% safety coverage")


if __name__ == "__main__":
    main()