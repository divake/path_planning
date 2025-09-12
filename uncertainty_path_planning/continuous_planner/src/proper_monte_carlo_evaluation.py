#!/usr/bin/env python3
"""
Proper Monte Carlo evaluation:
- Fix one environment
- Run MANY trials with different noise seeds
- Verify coverage guarantee properly
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
import hashlib
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from rrt_star_planner import RRTStar
from learnable_cp_final import FinalLearnableCP


def calculate_sample_size(confidence=0.95, margin_error=0.05, expected_prop=0.9):
    """
    Calculate required sample size for statistical significance
    Using Wilson score interval formula
    
    Args:
        confidence: Confidence level (0.95 for 95%)
        margin_error: Margin of error (0.05 for ±5%)
        expected_prop: Expected proportion (0.9 for 90% coverage)
    
    Returns:
        Required sample size
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    n = (z**2 * expected_prop * (1 - expected_prop)) / margin_error**2
    return int(np.ceil(n))


def generate_independent_seed(env_type: str, trial: int) -> int:
    """
    Generate truly independent seeds using hash
    
    Args:
        env_type: Environment type string
        trial: Trial number
        
    Returns:
        Independent seed
    """
    seed_str = f"{env_type}_{trial}_monte_carlo"
    hash_val = hashlib.sha256(seed_str.encode()).hexdigest()
    return int(hash_val[:8], 16) % 10000000


def monte_carlo_evaluation_single_env(env_type: str = "passages",
                                     num_trials: int = 1000,
                                     noise_level: float = 0.15,
                                     method: str = "naive"):
    """
    Proper Monte Carlo evaluation for ONE environment
    
    Args:
        env_type: Fixed environment type
        num_trials: Number of different noise realizations
        noise_level: Noise level (0.15 = 15%)
        method: "naive", "standard", or "learnable"
    
    Returns:
        Dictionary with results
    """
    
    print(f"\n{'='*70}")
    print(f"MONTE CARLO EVALUATION: {method.upper()} on {env_type.upper()}")
    print(f"Trials: {num_trials}, Noise: {noise_level*100:.0f}%")
    print(f"{'='*70}")
    
    # Create fixed environment
    env = ContinuousEnvironment(env_type=env_type)
    true_obstacles = env.obstacles
    
    # Initialize results
    results = {
        'collisions': 0,
        'paths_found': 0,
        'no_path': 0,
        'path_lengths': [],
        'collision_points': []
    }
    
    # Load model if needed
    if method in ["standard", "learnable"]:
        model = FinalLearnableCP(coverage=0.90, max_tau=1.0)
        # Simple tau for standard CP
        if method == "standard":
            tau = 0.5  # Fixed tau for standard CP
        
    # Run Monte Carlo trials
    for trial in range(num_trials):
        # Generate independent seed
        seed = generate_independent_seed(env_type, trial)
        
        # Add noise with unique seed
        perceived = ContinuousNoiseModel.add_thinning_noise(
            true_obstacles, thin_factor=noise_level, seed=seed
        )
        
        # Apply method-specific obstacle processing
        if method == "naive":
            planning_obstacles = perceived
        elif method == "standard":
            # Uniform inflation
            planning_obstacles = []
            for obs in perceived:
                planning_obstacles.append((
                    max(0, obs[0] - tau),
                    max(0, obs[1] - tau),
                    obs[2] + 2 * tau,
                    obs[3] + 2 * tau
                ))
        elif method == "learnable":
            # Adaptive inflation (simplified for demo)
            planning_obstacles = []
            for obs in perceived:
                # Simple adaptive tau based on obstacle size
                local_tau = 0.3 if obs[2] * obs[3] < 20 else 0.5
                planning_obstacles.append((
                    max(0, obs[0] - local_tau),
                    max(0, obs[1] - local_tau),
                    obs[2] + 2 * local_tau,
                    obs[3] + 2 * local_tau
                ))
        
        # Plan path
        planner = RRTStar((5, 15), (45, 15), planning_obstacles,
                         max_iter=500, robot_radius=0.5, seed=seed+1000)
        path = planner.plan()
        
        if path:
            results['paths_found'] += 1
            results['path_lengths'].append(planner.compute_path_length(path))
            
            # Check collision with TRUE obstacles
            collision_found = False
            for p in path:
                for obs in true_obstacles:
                    ox, oy, w, h = obs
                    # Check with robot radius
                    closest_x = max(ox, min(p[0], ox + w))
                    closest_y = max(oy, min(p[1], oy + h))
                    dist_sq = (p[0] - closest_x)**2 + (p[1] - closest_y)**2
                    
                    if dist_sq <= 0.5**2:  # Robot radius = 0.5
                        collision_found = True
                        results['collision_points'].append(p)
                        break
                
                if collision_found:
                    results['collisions'] += 1
                    break
        else:
            results['no_path'] += 1
        
        # Progress update
        if (trial + 1) % 100 == 0:
            current_rate = results['collisions'] / results['paths_found'] * 100 if results['paths_found'] > 0 else 0
            print(f"  Progress: {trial+1}/{num_trials} trials, "
                  f"collision rate: {current_rate:.1f}%")
    
    # Calculate statistics
    if results['paths_found'] > 0:
        collision_rate = results['collisions'] / results['paths_found']
        success_rate = 1 - collision_rate
        
        # Wilson confidence interval
        z = stats.norm.ppf(0.975)  # 95% confidence
        n = results['paths_found']
        p = success_rate
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        avg_length = np.mean(results['path_lengths'])
        std_length = np.std(results['path_lengths'])
    else:
        collision_rate = success_rate = 0
        ci_lower = ci_upper = 0
        avg_length = std_length = 0
    
    # Print results
    print(f"\n### RESULTS ###")
    print(f"Paths found: {results['paths_found']}/{num_trials}")
    print(f"Collisions: {results['collisions']}")
    print(f"Collision rate: {collision_rate*100:.1f}%")
    print(f"Success rate: {success_rate*100:.1f}% [{ci_lower*100:.1f}%-{ci_upper*100:.1f}%]")
    print(f"Avg path length: {avg_length:.1f} ± {std_length:.1f}")
    
    # Coverage guarantee check
    target_coverage = 0.90
    if method != "naive":
        if ci_lower >= target_coverage:
            print(f"✓ Coverage guarantee SATISFIED (>{target_coverage*100:.0f}%)")
        elif ci_upper >= target_coverage:
            print(f"⚠️ Coverage guarantee UNCERTAIN (CI includes {target_coverage*100:.0f}%)")
        else:
            print(f"✗ Coverage guarantee VIOLATED (<{target_coverage*100:.0f}%)")
    
    return {
        'collision_rate': collision_rate,
        'success_rate': success_rate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'avg_length': avg_length,
        'paths_found': results['paths_found'],
        'collisions': results['collisions']
    }


def compare_methods_monte_carlo():
    """Compare all methods with proper Monte Carlo"""
    
    # Calculate required sample size
    required_n = calculate_sample_size(confidence=0.95, margin_error=0.05, expected_prop=0.9)
    print(f"\nRequired sample size for 95% confidence, ±5% margin: {required_n}")
    
    # Use at least required sample size
    num_trials = max(200, required_n)  # Use 200 for speed, but note requirement
    
    results = {}
    
    for env_type in ['passages', 'open']:  # Skip narrow (too many failures)
        results[env_type] = {}
        
        for method in ['naive', 'standard']:
            result = monte_carlo_evaluation_single_env(
                env_type=env_type,
                num_trials=num_trials,
                noise_level=0.15,
                method=method
            )
            results[env_type][method] = result
    
    # Summary table
    print("\n" + "="*70)
    print("MONTE CARLO SUMMARY (Per Environment)")
    print("="*70)
    print(f"{'Environment':<12} {'Method':<12} {'Success Rate':<20} {'Coverage OK?':<15}")
    print("-"*70)
    
    for env_type in results:
        for method in results[env_type]:
            r = results[env_type][method]
            success_str = f"{r['success_rate']*100:.1f}% [{r['ci_lower']*100:.1f}-{r['ci_upper']*100:.1f}]"
            
            if method == 'naive':
                coverage_ok = "N/A"
            else:
                if r['ci_lower'] >= 0.90:
                    coverage_ok = "✓ Yes"
                elif r['ci_upper'] >= 0.90:
                    coverage_ok = "⚠️ Maybe"
                else:
                    coverage_ok = "✗ No"
            
            print(f"{env_type:<12} {method:<12} {success_str:<20} {coverage_ok:<15}")
    
    return results


def visualize_monte_carlo_distribution():
    """Visualize the distribution of outcomes across trials"""
    
    print("\n" + "="*70)
    print("VISUALIZING MONTE CARLO DISTRIBUTION")
    print("="*70)
    
    # Run 500 trials and collect per-batch statistics
    batch_size = 50
    num_batches = 10
    
    batch_rates = []
    
    for batch in range(num_batches):
        result = monte_carlo_evaluation_single_env(
            env_type="passages",
            num_trials=batch_size,
            noise_level=0.15,
            method="naive"
        )
        batch_rates.append(result['collision_rate'])
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(batch_rates, bins=10, density=True, alpha=0.7, 
            color='blue', edgecolor='black')
    plt.axvline(np.mean(batch_rates), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(batch_rates)*100:.1f}%')
    plt.xlabel('Collision Rate')
    plt.ylabel('Density')
    plt.title('Distribution of Collision Rates Across Batches (Naive Method)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('monte_carlo_distribution.png', dpi=150, bbox_inches='tight')
    print("  Distribution saved to monte_carlo_distribution.png")
    
    # Statistics
    print(f"\nBatch statistics ({batch_size} trials per batch):")
    print(f"  Mean collision rate: {np.mean(batch_rates)*100:.1f}%")
    print(f"  Std deviation: {np.std(batch_rates)*100:.1f}%")
    print(f"  Min: {np.min(batch_rates)*100:.1f}%")
    print(f"  Max: {np.max(batch_rates)*100:.1f}%")


if __name__ == "__main__":
    # Main evaluation
    results = compare_methods_monte_carlo()
    
    # Visualize distribution
    visualize_monte_carlo_distribution()
    
    # Final recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("1. Use at least 138 trials per environment for 95% confidence")
    print("2. Test each environment separately to verify coverage")
    print("3. Use hash-based seeds for true independence")
    print("4. Report confidence intervals, not just point estimates")
    print("5. Consider reducing noise to 10-15% for more realistic scenarios")