#!/usr/bin/env python3
"""
Quick Standard CP Test
Fast validation of core components without full path planning
"""

import numpy as np
import sys
import time

# Add current directory to path
sys.path.append('.')

from standard_cp_main import StandardCPPlanner
from standard_cp_noise_model import StandardCPNoiseModel
from standard_cp_nonconformity import StandardCPNonconformity


def test_quick_calibration():
    """Test calibration with mock data instead of actual path planning"""
    print("🚀 QUICK STANDARD CP TEST")
    print("="*60)
    
    try:
        # Initialize planner
        planner = StandardCPPlanner()
        
        # Override path planning to return mock data for speed
        original_plan = planner.plan_naive_path
        
        def mock_plan_naive_path(env_name, test_id, perceived_grid=None, seed=42):
            """Mock path planning that returns simple straight line"""
            # Create a simple straight-line path
            mock_path = [
                (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)
            ]
            return mock_path
        
        # Replace with mock for speed
        planner.plan_naive_path = mock_plan_naive_path
        
        print("✅ Initialized Standard CP planner")
        print(f"   Robot radius: {planner.config['robot']['radius']}m")
        print(f"   Fast mode: {planner.config['conformal_prediction']['calibration'].get('fast_mode', False)}")
        
        # Test calibration
        print("\n📊 Running quick calibration...")
        start_time = time.time()
        tau = planner.calibrate_global_tau()
        calibration_time = time.time() - start_time
        
        print(f"✅ Calibration completed in {calibration_time:.2f}s")
        print(f"   Global τ: {tau:.4f}m")
        
        # Check calibration data
        if planner.calibration_data:
            stats = planner.calibration_data.get('score_statistics', {})
            print(f"   Score statistics:")
            print(f"     Mean: {stats.get('mean', 0):.4f}m")
            print(f"     Std: {stats.get('std', 0):.4f}m")
            print(f"     Total trials: {planner.calibration_data.get('total_trials', 0)}")
            
            # Test CSV saving
            print(f"\n💾 Testing data saving...")
            planner.save_calibration_results()
            print(f"✅ Data saved successfully")
        
        # Test path planning comparison
        print(f"\n🛤️  Testing path planning comparison...")
        
        # Test naive planning
        naive_path = planner.plan_naive_path("office01add", 1, seed=42)
        print(f"   Naive path: {len(naive_path) if naive_path else 0} waypoints")
        
        # Test Standard CP planning (mock version)
        def mock_plan_with_standard_cp(env_name, test_id, perceived_grid=None, seed=42):
            # Return slightly longer path to simulate safety margin effect
            mock_path = [
                (0.0, 0.0), (0.8, 0.8), (1.6, 1.6), (2.4, 2.4), (3.2, 3.2), (4.0, 4.0)
            ]
            return mock_path
        
        planner.plan_with_standard_cp = mock_plan_with_standard_cp
        cp_path = planner.plan_with_standard_cp("office01add", 1, seed=42)
        print(f"   Standard CP path: {len(cp_path) if cp_path else 0} waypoints")
        
        # Test mini evaluation
        print(f"\n📈 Testing mini evaluation...")
        start_time = time.time()
        results = planner.evaluate_comparison(num_trials=4)  # Very small for speed
        eval_time = time.time() - start_time
        
        print(f"✅ Evaluation completed in {eval_time:.2f}s")
        
        # Print results
        for method in ['naive', 'standard_cp']:
            if method in results:
                data = results[method]
                print(f"   {method.replace('_', ' ').title()}:")
                print(f"     Success rate: {data.get('success_rate', 0)*100:.1f}%")
                print(f"     Avg planning time: {data.get('avg_planning_time', 0)*1000:.1f}ms")
                if data.get('avg_path_length', 0) > 0:
                    print(f"     Avg path length: {data.get('avg_path_length', 0):.1f}")
        
        # Test CSV data saving
        print(f"\n💾 Testing evaluation data saving...")
        planner.save_evaluation_results(results)
        print(f"✅ Evaluation data saved successfully")
        
        # Test results summary
        print(f"\n📋 Results Summary:")
        planner.print_results_summary()
        
        print(f"\n🎉 QUICK TEST COMPLETED SUCCESSFULLY!")
        print(f"   Total time: {time.time() - start_time:.2f}s")
        print(f"   τ value: {tau:.4f}m")
        print(f"   Ready for full evaluation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_outputs():
    """Test that all data outputs are being generated correctly"""
    print("\n🗂️  TESTING DATA OUTPUTS")
    print("="*40)
    
    import os
    
    # Check directory structure
    base_dir = "plots/standard_cp"
    subdirs = ["results", "plots", "data", "logs", "calibration", "evaluation"]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"✅ {subdir}: {len(files)} files")
            for file in files:
                print(f"   📄 {file}")
        else:
            print(f"❌ {subdir}: directory not found")
    
    return True


def main():
    """Run quick comprehensive test"""
    success = test_quick_calibration()
    if success:
        test_data_outputs()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)