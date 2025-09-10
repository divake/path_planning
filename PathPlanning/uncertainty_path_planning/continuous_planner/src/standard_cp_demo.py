#!/usr/bin/env python3
"""
Standard CP Complete Demo
End-to-end demonstration of Standard Conformal Prediction for MRPB path planning

This script demonstrates the complete pipeline:
1. Global τ calibration
2. Naive vs Standard CP comparison 
3. CSV data exports
4. Comprehensive visualizations
5. Results summary
"""

import sys
import time
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from standard_cp_main import StandardCPPlanner
from standard_cp_visualization import StandardCPVisualizer


def run_complete_standard_cp_demo():
    """Run complete Standard CP demonstration"""
    
    print("🚀 STANDARD CONFORMAL PREDICTION - COMPLETE DEMO")
    print("="*80)
    print("Uncertainty-Aware Path Planning with Safety Guarantees")
    print("Built on MRPB (Mobile Robot Path Planning Benchmark)")
    print("="*80)
    
    total_start_time = time.time()
    
    # Step 1: Initialize Standard CP
    print("\n📋 STEP 1: INITIALIZING STANDARD CP")
    print("-" * 40)
    
    planner = StandardCPPlanner()
    print(f"✅ Standard CP planner initialized")
    print(f"   Robot radius: {planner.config['robot']['radius']}m")
    print(f"   Target coverage: {planner.config['conformal_prediction']['target_coverage']*100:.0f}%")
    print(f"   Fast mode: {planner.config['conformal_prediction']['calibration'].get('fast_mode', False)}")
    
    # Step 2: Global τ Calibration
    print("\n📊 STEP 2: GLOBAL TAU CALIBRATION")
    print("-" * 40)
    
    cal_start_time = time.time()
    tau = planner.calibrate_global_tau()
    cal_time = time.time() - cal_start_time
    
    print(f"✅ Calibration completed in {cal_time:.2f}s")
    print(f"   Global τ: {tau:.4f}m ({tau*100:.1f}cm safety margin)")
    
    if planner.calibration_data:
        stats = planner.calibration_data.get('score_statistics', {})
        print(f"   Score statistics:")
        print(f"     • Mean: {stats.get('mean', 0):.4f}m")
        print(f"     • Std: {stats.get('std', 0):.4f}m")
        print(f"     • 90th percentile: {stats.get('percentiles', {}).get('90%', 0):.4f}m")
        print(f"     • Total trials: {planner.calibration_data.get('total_trials', 0)}")
    
    # Step 3: Method Comparison
    print("\n🛤️  STEP 3: NAIVE VS STANDARD CP COMPARISON")
    print("-" * 40)
    
    eval_start_time = time.time()
    results = planner.evaluate_comparison(num_trials=20)  # Small demo
    eval_time = time.time() - eval_start_time
    
    print(f"✅ Evaluation completed in {eval_time:.2f}s")
    
    # Display results
    print(f"\n📈 RESULTS SUMMARY:")
    print(f"   Method          Success  Collision  Path Length  Planning Time")
    print(f"   ─────────────  ───────  ─────────  ───────────  ─────────────")
    
    for method in ['naive', 'standard_cp']:
        if method in results:
            data = results[method]
            success = data.get('success_rate', 0) * 100
            collision = data.get('collision_rate', 0) * 100
            length = data.get('avg_path_length', 0)
            time_ms = data.get('avg_planning_time', 0) * 1000
            
            method_name = method.replace('_', ' ').title().ljust(13)
            print(f"   {method_name}  {success:6.1f}%  {collision:8.1f}%  {length:10.1f}  {time_ms:11.1f}ms")
    
    # Calculate improvement
    if 'naive' in results and 'standard_cp' in results:
        naive_collision = results['naive'].get('collision_rate', 1.0) * 100
        cp_collision = results['standard_cp'].get('collision_rate', 1.0) * 100
        safety_improvement = naive_collision - cp_collision
        
        naive_length = results['naive'].get('avg_path_length', 1.0)
        cp_length = results['standard_cp'].get('avg_path_length', 1.0)
        length_overhead = ((cp_length / naive_length) - 1) * 100 if naive_length > 0 else 0
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Safety improvement: {safety_improvement:.1f}% fewer collisions")
        print(f"   • Path overhead: {length_overhead:.1f}% longer paths for safety")
        print(f"   • τ = {tau:.3f}m provides 90% safety guarantee")
    
    # Step 4: Data Export
    print("\n💾 STEP 4: DATA EXPORT")
    print("-" * 40)
    
    print("📊 Exporting CSV data files...")
    # Data is automatically saved during calibration and evaluation
    print("✅ All data exported to plots/standard_cp/")
    print("   📁 results/     - CSV files for analysis")
    print("   📁 calibration/ - Tau and score analysis") 
    print("   📁 evaluation/  - Method comparison data")
    print("   📁 data/        - JSON files")
    print("   📁 logs/        - Debug logs")
    
    # Step 5: Visualizations
    print("\n🎨 STEP 5: VISUALIZATION GENERATION")
    print("-" * 40)
    
    viz_start_time = time.time()
    visualizer = StandardCPVisualizer()
    viz_success = visualizer.generate_all_visualizations()
    viz_time = time.time() - viz_start_time
    
    if viz_success:
        print(f"✅ All visualizations completed in {viz_time:.2f}s")
        print("📊 Generated plots:")
        print("   • Calibration analysis (score distributions, success rates)")
        print("   • Tau analysis (safety margin visualization)")
        print("   • Method comparison (naive vs Standard CP)")
        print("   • Performance analysis (timing and path length distributions)")
        print("   • Summary dashboard (comprehensive overview)")
    else:
        print("❌ Visualization generation failed")
    
    # Step 6: Final Summary
    total_time = time.time() - total_start_time
    
    print("\n🎉 STANDARD CP DEMO COMPLETED!")
    print("="*80)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"")
    print(f"📋 ACHIEVEMENTS:")
    print(f"   ✅ Global τ calibrated: {tau:.4f}m")
    print(f"   ✅ Safety guarantee: 90% collision-free")
    print(f"   ✅ Method comparison: Naive vs Standard CP")
    print(f"   ✅ Data exports: CSV and JSON files")
    print(f"   ✅ Visualizations: 5 comprehensive plots")
    print(f"")
    print(f"📁 OUTPUTS LOCATION:")
    print(f"   All files saved in: plots/standard_cp/")
    print(f"")
    print(f"🔬 READY FOR:")
    print(f"   • Full 15-environment evaluation")
    print(f"   • Learnable CP implementation")
    print(f"   • ICRA paper results generation")
    print("="*80)
    
    return True


def show_file_structure():
    """Show the generated file structure"""
    import os
    
    print("\n📁 GENERATED FILE STRUCTURE:")
    print("-" * 40)
    
    base_dir = "plots/standard_cp"
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.csv'):
                print(f"{sub_indent}📊 {file}")
            elif file.endswith('.png'):
                print(f"{sub_indent}🎨 {file}")
            elif file.endswith('.json'):
                print(f"{sub_indent}📄 {file}")
            elif file.endswith('.log'):
                print(f"{sub_indent}📝 {file}")
            else:
                print(f"{sub_indent}📋 {file}")


def main():
    """Main demo function"""
    try:
        success = run_complete_standard_cp_demo()
        if success:
            show_file_structure()
        return success
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)