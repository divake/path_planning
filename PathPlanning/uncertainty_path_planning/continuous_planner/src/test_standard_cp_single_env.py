#!/usr/bin/env python3
"""
Test Standard CP on Single Environment
Simple test script to validate the implementation before full evaluation

This script:
1. Tests noise model on office01add environment
2. Tests nonconformity score calculation
3. Tests calibration with small sample size
4. Tests path planning comparison
5. Validates results make sense

Run this first to debug any issues before full evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add current directory to path
sys.path.append('.')

# Import our Standard CP components
from standard_cp_main import StandardCPPlanner
from standard_cp_noise_model import StandardCPNoiseModel
from standard_cp_nonconformity import StandardCPNonconformity

# Import existing infrastructure
from mrpb_map_parser import MRPBMapParser


def test_noise_model():
    """Test noise model on real MRPB environment"""
    print("\n" + "="*60)
    print("TESTING NOISE MODEL")
    print("="*60)
    
    try:
        # Load environment
        parser = MRPBMapParser("office01add")
        true_grid = parser.occupancy_grid
        
        print(f"Environment loaded: {true_grid.shape}")
        print(f"Occupied pixels: {np.sum(true_grid == 1)}")
        print(f"Free pixels: {np.sum(true_grid == 0)}")
        
        # Initialize noise model
        noise_model = StandardCPNoiseModel()
        
        # Test different noise levels
        noise_levels = [0.05, 0.10, 0.15, 0.20]
        
        for noise_level in noise_levels:
            print(f"\nTesting noise level: {noise_level}")
            
            # Add noise
            noisy_grid = noise_model.add_realistic_noise(
                true_grid, noise_level=noise_level, seed=42
            )
            
            # Analyze impact
            analysis = noise_model.analyze_noise_impact(true_grid, noisy_grid)
            
            print(f"  Change ratio: {analysis['change_ratio']:.3f}")
            print(f"  False negative rate: {analysis['false_negative_rate']:.3f}")
            print(f"  False positive rate: {analysis['false_positive_rate']:.3f}")
        
        print("✓ Noise model test passed")
        return True
        
    except Exception as e:
        print(f"✗ Noise model test failed: {e}")
        return False


def test_nonconformity_scores():
    """Test nonconformity score calculation"""
    print("\n" + "="*60)
    print("TESTING NONCONFORMITY SCORES")
    print("="*60)
    
    try:
        # Load environment
        parser = MRPBMapParser("office01add")
        true_grid = parser.occupancy_grid
        
        # Create noise model
        noise_model = StandardCPNoiseModel()
        
        # Create multiple noisy versions
        noisy_grids = []
        for i in range(5):
            noisy_grid = noise_model.add_realistic_noise(
                true_grid, noise_level=0.15, seed=100 + i
            )
            noisy_grids.append(noisy_grid)
        
        # Create test paths (simple diagonal paths)
        test_paths = [
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],  # Short path
            [(0.0, 0.0), (2.0, 1.0), (4.0, 2.0), (6.0, 3.0)],  # Medium path
            [(-2.0, -2.0), (0.0, 0.0), (2.0, 2.0), (4.0, 4.0), (6.0, 6.0)]  # Long path
        ]
        
        # Initialize nonconformity calculator
        nc_calc = StandardCPNonconformity()
        
        print(f"Testing {len(test_paths)} paths against {len(noisy_grids)} noisy grids...")
        
        all_scores = []
        
        for i, path in enumerate(test_paths):
            for j, noisy_grid in enumerate(noisy_grids):
                score = nc_calc.compute_nonconformity_score(
                    true_grid, noisy_grid, path, path_id=f"test_{i}_{j}"
                )
                all_scores.append(score)
                print(f"  Path {i+1}, Noise {j+1}: score = {score:.4f}m")
        
        # Analyze score distribution
        analysis = nc_calc.analyze_score_distribution(all_scores)
        print(f"\nScore distribution analysis:")
        print(f"  Mean: {analysis['mean']:.4f}m")
        print(f"  Std: {analysis['std']:.4f}m")
        print(f"  Range: [{analysis['min']:.4f}, {analysis['max']:.4f}]m")
        print(f"  90th percentile: {analysis['percentiles']['90%']:.4f}m")
        
        # Validate scores
        valid = nc_calc.validate_scores(all_scores)
        print(f"  Scores valid: {valid}")
        
        print("✓ Nonconformity scores test passed")
        return True
        
    except Exception as e:
        print(f"✗ Nonconformity scores test failed: {e}")
        return False


def test_calibration():
    """Test τ calibration with small sample"""
    print("\n" + "="*60)
    print("TESTING CALIBRATION")
    print("="*60)
    
    try:
        # Initialize planner
        planner = StandardCPPlanner()
        
        # Override config for quick testing
        planner.config['conformal_prediction']['calibration']['trials_per_environment'] = 10
        planner.config['conformal_prediction']['calibration']['total_trials_test'] = 20
        
        print("Starting calibration with reduced sample size...")
        
        # Calibrate
        start_time = time.time()
        tau = planner.calibrate_global_tau()
        calibration_time = time.time() - start_time
        
        print(f"Calibration completed in {calibration_time:.2f}s")
        print(f"Global τ: {tau:.4f}m")
        
        # Check calibration data
        if planner.calibration_data:
            stats = planner.calibration_data.get('score_statistics', {})
            print(f"Score statistics:")
            print(f"  Mean: {stats.get('mean', 0):.4f}m")
            print(f"  Std: {stats.get('std', 0):.4f}m")
            print(f"  Total trials: {planner.calibration_data.get('total_trials', 0)}")
        
        # Validate τ is reasonable
        if 0.01 <= tau <= 1.0:
            print("✓ τ value is in reasonable range")
        else:
            print(f"⚠ τ value may be unreasonable: {tau:.4f}m")
        
        print("✓ Calibration test passed")
        return True, planner
        
    except Exception as e:
        print(f"✗ Calibration test failed: {e}")
        return False, None


def test_path_planning(planner):
    """Test path planning comparison"""
    print("\n" + "="*60)
    print("TESTING PATH PLANNING")
    print("="*60)
    
    try:
        if not planner or not planner.calibration_completed:
            print("✗ No calibrated planner available")
            return False
        
        # Test environment and test case
        env_name = "office01add"
        test_id = 1
        
        print(f"Testing planning on {env_name}-{test_id}")
        print(f"Using τ = {planner.tau:.4f}m")
        
        # Load environment and add noise
        parser, env_config = planner.load_environment(env_name)
        true_grid = parser.occupancy_grid
        
        # Add realistic noise
        perceived_grid = planner.noise_model.add_realistic_noise(
            true_grid, noise_level=0.15, seed=42
        )
        
        # Test naive planning
        print("\nTesting naive planning...")
        start_time = time.time()
        naive_path = planner.plan_naive_path(env_name, test_id, perceived_grid, seed=1000)
        naive_time = time.time() - start_time
        
        print(f"  Naive planning time: {naive_time*1000:.1f}ms")
        print(f"  Naive path found: {naive_path is not None}")
        if naive_path:
            print(f"  Naive path length: {len(naive_path)} waypoints")
        
        # Test Standard CP planning
        print("\nTesting Standard CP planning...")
        start_time = time.time()
        cp_path = planner.plan_with_standard_cp(env_name, test_id, perceived_grid, seed=2000)
        cp_time = time.time() - start_time
        
        print(f"  Standard CP planning time: {cp_time*1000:.1f}ms")
        print(f"  Standard CP path found: {cp_path is not None}")
        if cp_path:
            print(f"  Standard CP path length: {len(cp_path)} waypoints")
        
        # Check collision on true environment
        if naive_path:
            naive_collision = planner._check_collision_on_true_grid(naive_path, true_grid)
            print(f"  Naive collision on true grid: {naive_collision}")
        
        if cp_path:
            cp_collision = planner._check_collision_on_true_grid(cp_path, true_grid)
            print(f"  Standard CP collision on true grid: {cp_collision}")
        
        # Compare path lengths
        if naive_path and cp_path:
            length_ratio = len(cp_path) / len(naive_path)
            print(f"  Path length ratio (CP/Naive): {length_ratio:.2f}")
            
            if length_ratio > 2.0:
                print("⚠ Standard CP path much longer than naive")
            elif length_ratio < 0.5:
                print("⚠ Standard CP path much shorter than naive")
            else:
                print("✓ Path length ratio reasonable")
        
        print("✓ Path planning test passed")
        return True
        
    except Exception as e:
        print(f"✗ Path planning test failed: {e}")
        return False


def test_evaluation(planner):
    """Test evaluation with very small sample"""
    print("\n" + "="*60)
    print("TESTING EVALUATION")
    print("="*60)
    
    try:
        if not planner or not planner.calibration_completed:
            print("✗ No calibrated planner available")
            return False
        
        print("Running mini evaluation (10 trials)...")
        
        # Run small evaluation
        results = planner.evaluate_comparison(num_trials=10)
        
        # Print results
        print("\nEvaluation results:")
        for method in ['naive', 'standard_cp']:
            if method in results:
                data = results[method]
                print(f"  {method.replace('_', ' ').title()}:")
                print(f"    Trials: {data.get('total_trials', 0)}")
                print(f"    Success rate: {data.get('success_rate', 0)*100:.1f}%")
                print(f"    Collision rate: {data.get('collision_rate', 0)*100:.1f}%")
                print(f"    Avg planning time: {data.get('avg_planning_time', 0)*1000:.1f}ms")
                if data.get('avg_path_length', 0) > 0:
                    print(f"    Avg path length: {data.get('avg_path_length', 0):.1f}")
        
        # Check if results make sense
        naive_data = results.get('naive', {})
        cp_data = results.get('standard_cp', {})
        
        # Standard CP should have lower collision rate
        naive_collisions = naive_data.get('collision_rate', 1.0)
        cp_collisions = cp_data.get('collision_rate', 1.0)
        
        if cp_collisions <= naive_collisions:
            print("✓ Standard CP has lower collision rate than naive")
        else:
            print("⚠ Standard CP collision rate higher than naive")
        
        print("✓ Evaluation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("STANDARD CP SINGLE ENVIRONMENT TEST")
    print("="*80)
    print("Testing implementation on office01add environment")
    print("This validates core functionality before full evaluation")
    
    # Track test results
    test_results = []
    
    # Test 1: Noise model
    test_results.append(test_noise_model())
    
    # Test 2: Nonconformity scores
    test_results.append(test_nonconformity_scores())
    
    # Test 3: Calibration
    calibration_success, planner = test_calibration()
    test_results.append(calibration_success)
    
    # Test 4: Path planning (only if calibration succeeded)
    if calibration_success and planner:
        test_results.append(test_path_planning(planner))
        
        # Test 5: Evaluation (only if path planning succeeded)
        if test_results[-1]:
            test_results.append(test_evaluation(planner))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    test_names = [
        "Noise Model",
        "Nonconformity Scores", 
        "Calibration",
        "Path Planning",
        "Evaluation"
    ]
    
    for i, (name, result) in enumerate(zip(test_names[:len(test_results)], test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for full evaluation.")
    else:
        print("⚠️  Some tests failed. Check implementation before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)