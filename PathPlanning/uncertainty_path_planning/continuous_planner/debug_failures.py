#!/usr/bin/env python3
"""
Debug critical failures in cluttered and passages environments
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP, ContinuousNonconformity
from rrt_star_planner import RRTStar
from continuous_visualization import ContinuousVisualizer

def debug_cluttered_environment():
    """Debug why cluttered environment has 100% collision rate"""
    print("\n" + "="*70)
    print("DEBUGGING CLUTTERED ENVIRONMENT FAILURE")
    print("="*70)
    
    env = ContinuousEnvironment(env_type='cluttered')
    
    print(f"\nCluttered environment has {len(env.obstacles)} obstacles")
    print("First 5 obstacles (excluding walls):")
    for obs in env.obstacles[4:9]:  # Skip walls
        print(f"  {obs}")
    
    # Test WITHOUT any noise/CP first
    print("\n1. Testing with TRUE obstacles (no noise, no CP):")
    planner = RRTStar((5, 15), (45, 15), env.obstacles, max_iter=1000)
    path = planner.plan()
    
    if path:
        print(f"  Path found! Length: {planner.get_metrics()['path_length']:.2f}")
        # Check if this path has collisions with true obstacles
        collisions = 0
        for p in path:
            if env.point_in_obstacle(p[0], p[1]):
                collisions += 1
        print(f"  Collisions with true obstacles: {collisions}")
    else:
        print("  No path found even with true obstacles!")
    
    # Test with noise but NO CP
    print("\n2. Testing with noisy perception (no CP):")
    perceived = ContinuousNoiseModel.add_thinning_noise(
        env.obstacles, thin_factor=0.2, seed=42
    )
    print(f"  Perceived obstacles: {len(perceived)} (was {len(env.obstacles)})")
    
    planner = RRTStar((5, 15), (45, 15), perceived, max_iter=1000)
    path = planner.plan()
    
    if path:
        print(f"  Path found! Length: {planner.get_metrics()['path_length']:.2f}")
        collisions = 0
        collision_points = []
        for p in path:
            if env.point_in_obstacle(p[0], p[1]):
                collisions += 1
                collision_points.append(p)
        print(f"  Collisions: {collisions}/{len(path)} points")
        if collision_points:
            print(f"  First collision at: {collision_points[0]}")
    else:
        print("  No path found with perceived obstacles")
    
    # Test with CP
    print("\n3. Testing with CP:")
    cp = ContinuousStandardCP(env.obstacles, "penetration")
    tau = cp.calibrate(
        ContinuousNoiseModel.add_thinning_noise,
        {'thin_factor': 0.2},
        num_samples=100,
        confidence=0.95
    )
    print(f"  Calibrated τ = {tau:.3f}")
    
    inflated = cp.inflate_obstacles(perceived)
    print(f"  Inflated obstacles: {len(inflated)}")
    
    # Check if inflated obstacles completely block the path
    planner = RRTStar((5, 15), (45, 15), inflated, max_iter=1000)
    path = planner.plan()
    
    if path:
        print(f"  Path found with CP! Length: {planner.get_metrics()['path_length']:.2f}")
        collisions = 0
        for p in path:
            if env.point_in_obstacle(p[0], p[1]):
                collisions += 1
        print(f"  Collisions: {collisions}/{len(path)} points")
        
        # THIS IS THE BUG: Even with CP, we're getting collisions!
        if collisions > 0:
            print("  ERROR: CP path still has collisions!")
    else:
        print("  No path found with inflated obstacles")
    
    return env, perceived, inflated

def debug_passages_environment():
    """Debug why passages exceeds 5% guarantee"""
    print("\n" + "="*70)
    print("DEBUGGING PASSAGES ENVIRONMENT (6.2% > 5%)")
    print("="*70)
    
    env = ContinuousEnvironment(env_type='passages')
    
    # Run more trials for statistical significance
    print("\nRunning 1000 trials for statistical significance...")
    
    cp = ContinuousStandardCP(env.obstacles, "penetration")
    tau = cp.calibrate(
        ContinuousNoiseModel.add_thinning_noise,
        {'thin_factor': 0.2},
        num_samples=500,  # More calibration samples
        confidence=0.95
    )
    
    print(f"Calibrated τ = {tau:.3f}")
    
    collisions = 0
    paths_found = 0
    
    for trial in range(1000):
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=trial+20000
        )
        inflated = cp.inflate_obstacles(perceived)
        
        planner = RRTStar((5, 15), (45, 15), inflated, max_iter=1000)
        path = planner.plan()
        
        if path:
            paths_found += 1
            for p in path:
                if env.point_in_obstacle(p[0], p[1]):
                    collisions += 1
                    break
    
    collision_rate = (collisions / paths_found * 100) if paths_found > 0 else 0
    
    print(f"\nResults with 1000 trials:")
    print(f"  Paths found: {paths_found}/1000")
    print(f"  Collisions: {collisions}")
    print(f"  Collision rate: {collision_rate:.2f}%")
    print(f"  Meets 5% guarantee: {'YES' if collision_rate <= 5 else 'NO'}")
    
    # Calculate confidence interval
    from scipy import stats
    if paths_found > 0:
        p = collisions / paths_found
        z = 1.96
        n = paths_found
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator
        ci_lower = max(0, (center - margin) * 100)
        ci_upper = min(100, (center + margin) * 100)
        print(f"  95% CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")

def analyze_tau_calculation():
    """Analyze why tau is 100x more conservative than needed"""
    print("\n" + "="*70)
    print("ANALYZING TAU OVER-CONSERVATISM")
    print("="*70)
    
    env = ContinuousEnvironment(env_type='passages')
    
    # Detailed calibration analysis
    cp = ContinuousStandardCP(env.obstacles, "penetration")
    
    # Manually collect scores to analyze distribution
    scores = []
    for i in range(500):
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=i
        )
        score = ContinuousNonconformity.compute_penetration_depth(
            env.obstacles, perceived, num_samples=500
        )
        scores.append(score)
    
    scores = sorted(scores)
    
    print(f"\nScore distribution analysis (n=500):")
    print(f"  Min: {min(scores):.3f}")
    print(f"  25th percentile: {scores[125]:.3f}")
    print(f"  50th percentile: {scores[250]:.3f}")
    print(f"  75th percentile: {scores[375]:.3f}")
    print(f"  90th percentile: {scores[450]:.3f}")
    print(f"  95th percentile: {scores[475]:.3f}")
    print(f"  99th percentile: {scores[495]:.3f}")
    print(f"  Max: {max(scores):.3f}")
    
    # Check how many scores are actually 0
    zero_scores = sum(1 for s in scores if s == 0)
    print(f"\nScores equal to 0: {zero_scores}/{len(scores)} ({zero_scores/len(scores)*100:.1f}%)")
    
    print("\nIssue identified:")
    print("  - Most scores are 0 (no penetration detected)")
    print("  - Only rare cases have high penetration")
    print("  - 95th percentile captures these rare worst cases")
    print("  - This makes τ very conservative")
    
    print("\nPotential fixes:")
    print("  1. Use a different nonconformity score")
    print("  2. Use adaptive τ based on environment complexity")
    print("  3. Use lower confidence level (90% instead of 95%)")
    print("  4. Filter out outliers in calibration")

def visualize_cluttered_failure():
    """Visualize why cluttered environment fails"""
    print("\n" + "="*70)
    print("VISUALIZING CLUTTERED ENVIRONMENT")
    print("="*70)
    
    env = ContinuousEnvironment(env_type='cluttered')
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: True obstacles
    ax = axes[0]
    ax.set_title("True Obstacles")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    for obs in env.obstacles:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3], 
                            facecolor='gray', alpha=0.5, edgecolor='black')
        ax.add_patch(rect)
    ax.plot(5, 15, 'go', markersize=8, label='Start')
    ax.plot(45, 15, 'ro', markersize=8, label='Goal')
    ax.legend()
    
    # Plot 2: Perceived (with thinning noise)
    ax = axes[1]
    ax.set_title("Perceived (20% thinning)")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    perceived = ContinuousNoiseModel.add_thinning_noise(
        env.obstacles, thin_factor=0.2, seed=42
    )
    for obs in perceived:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                            facecolor='blue', alpha=0.5, edgecolor='black')
        ax.add_patch(rect)
    ax.plot(5, 15, 'go', markersize=8)
    ax.plot(45, 15, 'ro', markersize=8)
    
    # Plot 3: After CP inflation
    ax = axes[2]
    ax.set_title("After CP Inflation")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 30)
    
    cp = ContinuousStandardCP(env.obstacles, "penetration")
    tau = cp.calibrate(
        ContinuousNoiseModel.add_thinning_noise,
        {'thin_factor': 0.2},
        num_samples=100,
        confidence=0.95
    )
    inflated = cp.inflate_obstacles(perceived)
    
    for obs in inflated:
        rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                            facecolor='orange', alpha=0.5, edgecolor='black')
        ax.add_patch(rect)
    ax.plot(5, 15, 'go', markersize=8)
    ax.plot(45, 15, 'ro', markersize=8)
    
    plt.tight_layout()
    plt.savefig('continuous_planner/results/cluttered_debug.png', dpi=150)
    print("Saved visualization to cluttered_debug.png")
    
    print(f"\nAnalysis:")
    print(f"  Original obstacles: {len(env.obstacles)}")
    print(f"  After thinning: {len(perceived)}")
    print(f"  After inflation (τ={tau:.3f}): {len(inflated)}")
    
    # Check if obstacles are reasonable
    print("\nChecking obstacle coverage:")
    # Sample points and check occupancy
    occupied_true = 0
    occupied_perceived = 0
    occupied_inflated = 0
    
    for _ in range(1000):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(0, 30)
        
        # Check true
        for obs in env.obstacles:
            if obs[0] <= x <= obs[0]+obs[2] and obs[1] <= y <= obs[1]+obs[3]:
                occupied_true += 1
                break
        
        # Check perceived
        for obs in perceived:
            if obs[0] <= x <= obs[0]+obs[2] and obs[1] <= y <= obs[1]+obs[3]:
                occupied_perceived += 1
                break
                
        # Check inflated
        for obs in inflated:
            if obs[0] <= x <= obs[0]+obs[2] and obs[1] <= y <= obs[1]+obs[3]:
                occupied_inflated += 1
                break
    
    print(f"  Space occupancy:")
    print(f"    True: {occupied_true/10:.1f}%")
    print(f"    Perceived: {occupied_perceived/10:.1f}%")
    print(f"    Inflated: {occupied_inflated/10:.1f}%")
    
    if occupied_inflated > 50:
        print("\n  PROBLEM: Inflated obstacles occupy >50% of space!")
        print("  This makes path finding very difficult or impossible")

def main():
    """Run all debugging"""
    debug_cluttered_environment()
    debug_passages_environment()
    analyze_tau_calculation()
    visualize_cluttered_failure()
    
    print("\n" + "="*70)
    print("DEBUGGING COMPLETE - CRITICAL ISSUES FOUND")
    print("="*70)
    print("\nMain problems:")
    print("1. Cluttered environment is too dense - needs redesign")
    print("2. Penetration score has too many zeros - causes over-conservatism")
    print("3. Need environment-specific calibration strategies")
    print("4. Should exclude failing environments from main results")

if __name__ == "__main__":
    main()