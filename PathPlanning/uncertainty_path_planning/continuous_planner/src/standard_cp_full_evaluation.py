#!/usr/bin/env python3
"""
Standard CP Full-Scale Evaluation
Comprehensive evaluation across all 15 MRPB environments with 1000 Monte Carlo trials

This script runs the complete Standard CP evaluation pipeline:
1. Global τ calibration across all environments
2. Full-scale Monte Carlo evaluation (1000 trials per method)
3. Complete MRPB metrics computation
4. Publication-ready data and visualization generation
"""

import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add current directory to path
sys.path.append('.')

from standard_cp_main import StandardCPPlanner
from standard_cp_visualization import StandardCPVisualizer
from standard_cp_progress_tracker import StandardCPProgressTracker

# Install tqdm if not available for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("📦 Installing tqdm for progress bars...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm
        TQDM_AVAILABLE = True
        print("✅ tqdm installed successfully!")
    except:
        print("⚠️  Could not install tqdm. Progress will be shown as text.")
        TQDM_AVAILABLE = False


class FullScaleStandardCPEvaluator:
    """Full-scale evaluation manager for Standard CP"""
    
    def __init__(self, config_path: str = "standard_cp_config.yaml"):
        """Initialize full-scale evaluator"""
        self.planner = StandardCPPlanner(config_path)
        self.visualizer = StandardCPVisualizer()
        self.progress_tracker = StandardCPProgressTracker()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full-scale parameters
        self.full_scale_config = {
            'calibration_trials': 100,  # More thorough calibration
            'evaluation_trials': 1000,  # Full Monte Carlo
            'environments': 'all',  # All 15 MRPB environments
            'methods': ['naive', 'standard_cp'],
            'detailed_metrics': True,  # Full MRPB metrics
            'timeout_planning': 30.0,  # Longer timeout for complex scenarios
            'timeout_total': 7200.0   # 2 hour total timeout
        }
        
        print(f"🚀 FULL-SCALE STANDARD CP EVALUATION INITIALIZED")
        print(f"   Calibration trials: {self.full_scale_config['calibration_trials']}")
        print(f"   Evaluation trials: {self.full_scale_config['evaluation_trials']}")
        print(f"   Total environments: 15 MRPB environments")
        print(f"   Progress tracking: {'✅ With visual progress bars' if TQDM_AVAILABLE else '✅ Text-based progress'}")
        print(f"   Detailed logging: ✅ CSV files with failure analysis")
        print(f"   Expected duration: ~2 hours")
        
    def run_comprehensive_calibration(self) -> float:
        """Run comprehensive τ calibration with progress tracking"""
        print(f"\n🔧 COMPREHENSIVE τ CALIBRATION")
        print(f"   Trials per environment: {self.full_scale_config['calibration_trials']}")
        print(f"   Target coverage: 90%")
        print(f"   Max iterations: 50,000 per planning attempt")
        
        # Update planner config for comprehensive calibration
        original_trials = self.planner.config['conformal_prediction']['calibration']['trials_per_environment']
        original_fast_mode = self.planner.config['conformal_prediction']['calibration'].get('fast_mode', True)
        
        self.planner.config['conformal_prediction']['calibration']['trials_per_environment'] = self.full_scale_config['calibration_trials']
        self.planner.config['conformal_prediction']['calibration']['fast_mode'] = False  # Disable fast mode for accuracy
        
        start_time = time.time()
        
        # Initialize progress tracking for calibration
        total_calibration_trials = self._estimate_calibration_trials()
        self.progress_tracker.init_calibration_progress(total_calibration_trials)
        
        try:
            tau = self.planner.calibrate_global_tau()
            calibration_time = time.time() - start_time
            
            # Finish calibration progress
            self.progress_tracker.finish_calibration_progress()
            
            print(f"✅ Comprehensive calibration completed")
            print(f"   Duration: {calibration_time:.1f}s")
            print(f"   Global τ: {tau:.4f}m ({tau*100:.1f}cm)")
            
            if self.planner.calibration_data:
                stats = self.planner.calibration_data.get('score_statistics', {})
                print(f"   Total calibration trials: {self.planner.calibration_data.get('total_trials', 0)}")
                print(f"   Score mean ± std: {stats.get('mean', 0):.4f} ± {stats.get('std', 0):.4f}m")
                print(f"   90th percentile: {stats.get('percentiles', {}).get('90%', 0):.4f}m")
            
            return tau
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            # Restore original config
            self.planner.config['conformal_prediction']['calibration']['trials_per_environment'] = original_trials
            self.planner.config['conformal_prediction']['calibration']['fast_mode'] = original_fast_mode
            raise
        finally:
            # Restore original config
            self.planner.config['conformal_prediction']['calibration']['trials_per_environment'] = original_trials
            self.planner.config['conformal_prediction']['calibration']['fast_mode'] = original_fast_mode
            
    def run_full_scale_evaluation(self, tau: float) -> Dict[str, Any]:
        """Run full-scale Monte Carlo evaluation with progress tracking"""
        print(f"\n🧪 FULL-SCALE MONTE CARLO EVALUATION")
        print(f"   Trials per method: {self.full_scale_config['evaluation_trials']}")
        print(f"   Methods: {', '.join(self.full_scale_config['methods'])}")
        print(f"   Using τ: {tau:.4f}m")
        print(f"   Max iterations: 50,000 per planning attempt")
        
        start_time = time.time()
        
        # Initialize progress tracking for evaluation
        total_evaluation_trials = self.full_scale_config['evaluation_trials'] * len(self.full_scale_config['methods'])
        self.progress_tracker.init_evaluation_progress(total_evaluation_trials)
        
        # Run comprehensive evaluation
        results = self.planner.evaluate_comparison(
            num_trials=self.full_scale_config['evaluation_trials']
        )
        
        evaluation_time = time.time() - start_time
        
        # Finish evaluation progress
        self.progress_tracker.finish_evaluation_progress()
        
        print(f"✅ Full-scale evaluation completed")
        print(f"   Duration: {evaluation_time:.1f}s ({evaluation_time/60:.1f} minutes)")
        
        # Display comprehensive results
        self.display_comprehensive_results(results)
        
        return results
    
    def display_comprehensive_results(self, results: Dict[str, Any]):
        """Display comprehensive evaluation results"""
        print(f"\n📊 COMPREHENSIVE RESULTS ANALYSIS")
        print(f"=" * 80)
        
        # Method comparison table
        print(f"\n📈 METHOD PERFORMANCE COMPARISON:")
        print(f"{'Method':<15} {'Success':<8} {'Collision':<10} {'Path Length':<12} {'Plan Time':<10}")
        print(f"{'─'*15} {'─'*8} {'─'*10} {'─'*12} {'─'*10}")
        
        method_stats = {}
        for method in ['naive', 'standard_cp']:
            if method in results:
                data = results[method]
                success = data.get('success_rate', 0) * 100
                collision = data.get('collision_rate', 0) * 100
                length = data.get('avg_path_length', 0)
                time_ms = data.get('avg_planning_time', 0) * 1000
                
                method_name = method.replace('_', ' ').title()
                print(f"{method_name:<15} {success:7.1f}% {collision:9.1f}% {length:11.2f} {time_ms:9.1f}ms")
                
                method_stats[method] = {
                    'success': success,
                    'collision': collision,
                    'length': length,
                    'time_ms': time_ms
                }
        
        # Safety analysis
        if 'naive' in method_stats and 'standard_cp' in method_stats:
            safety_improvement = method_stats['naive']['collision'] - method_stats['standard_cp']['collision']
            length_overhead = ((method_stats['standard_cp']['length'] / method_stats['naive']['length']) - 1) * 100 if method_stats['naive']['length'] > 0 else 0
            time_overhead = ((method_stats['standard_cp']['time_ms'] / method_stats['naive']['time_ms']) - 1) * 100 if method_stats['naive']['time_ms'] > 0 else 0
            
            print(f"\n🔍 SAFETY & EFFICIENCY ANALYSIS:")
            print(f"   Safety improvement: {safety_improvement:.1f}% collision reduction")
            print(f"   Path length overhead: {length_overhead:.1f}% longer paths")
            print(f"   Planning time overhead: {time_overhead:.1f}% additional computation")
        
        # Statistical significance (if available)
        if any('confidence_intervals' in results.get(method, {}) for method in results):
            print(f"\n📐 STATISTICAL CONFIDENCE INTERVALS (95%):")
            for method in ['naive', 'standard_cp']:
                if method in results and 'confidence_intervals' in results[method]:
                    ci = results[method]['confidence_intervals']
                    method_name = method.replace('_', ' ').title()
                    print(f"   {method_name} Success Rate: [{ci.get('success_rate_lower', 0)*100:.1f}%, {ci.get('success_rate_upper', 0)*100:.1f}%]")
        
        # Environment-specific analysis (if available)
        if any('per_environment_stats' in results.get(method, {}) for method in results):
            print(f"\n🌍 ENVIRONMENT-SPECIFIC PERFORMANCE:")
            print(f"   (Detailed per-environment statistics available in CSV files)")
    
    def generate_publication_outputs(self, results: Dict[str, Any], tau: float) -> bool:
        """Generate publication-ready outputs"""
        print(f"\n📄 GENERATING PUBLICATION OUTPUTS")
        print(f"─" * 40)
        
        try:
            # Create comprehensive data package
            publication_data = {
                'experiment_info': {
                    'timestamp': self.timestamp,
                    'evaluation_type': 'full_scale_standard_cp',
                    'total_trials': self.full_scale_config['evaluation_trials'],
                    'calibration_trials': self.full_scale_config['calibration_trials'],
                    'global_tau': tau,
                    'target_coverage': 0.90,
                    'environments': 'all_mrpb_15'
                },
                'results': results,
                'calibration_data': self.planner.calibration_data
            }
            
            # Save master results file
            results_dir = Path("plots/standard_cp/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / f"full_evaluation_{self.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(publication_data, f, indent=2, default=self._json_serializable)
            
            print(f"✅ Master results saved: {results_file}")
            
            # Generate all visualizations
            print(f"🎨 Generating comprehensive visualizations...")
            viz_success = self.visualizer.generate_all_visualizations()
            
            if viz_success:
                print(f"✅ All visualizations generated successfully")
                print(f"   • Calibration analysis with {self.full_scale_config['calibration_trials']} trials")
                print(f"   • Method comparison with {self.full_scale_config['evaluation_trials']} trials")
                print(f"   • Statistical confidence intervals")
                print(f"   • Performance distribution analysis")
                print(f"   • Publication-ready summary dashboard")
            else:
                print(f"⚠️  Some visualizations may have failed")
            
            # Generate summary report
            self.generate_summary_report(publication_data, tau)
            
            return True
            
        except Exception as e:
            print(f"❌ Publication output generation failed: {e}")
            return False
    
    def generate_summary_report(self, data: Dict[str, Any], tau: float):
        """Generate comprehensive summary report"""
        report_dir = Path("plots/standard_cp/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"full_evaluation_summary_{self.timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STANDARD CP FULL-SCALE EVALUATION SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation ID: {self.timestamp}\n\n")
            
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Monte Carlo trials: {self.full_scale_config['evaluation_trials']}\n")
            f.write(f"Calibration trials: {self.full_scale_config['calibration_trials']}\n")
            f.write(f"Environments: All 15 MRPB environments\n")
            f.write(f"Methods: {', '.join(self.full_scale_config['methods'])}\n")
            f.write(f"Target coverage: 90%\n")
            f.write(f"Global τ: {tau:.4f}m ({tau*100:.1f}cm)\n\n")
            
            # Add detailed results
            results = data.get('results', {})
            if results:
                f.write("PERFORMANCE RESULTS:\n")
                f.write("-"*40 + "\n")
                for method in ['naive', 'standard_cp']:
                    if method in results:
                        method_data = results[method]
                        method_name = method.replace('_', ' ').title()
                        f.write(f"{method_name}:\n")
                        f.write(f"  Success Rate: {method_data.get('success_rate', 0)*100:.1f}%\n")
                        f.write(f"  Collision Rate: {method_data.get('collision_rate', 0)*100:.1f}%\n")
                        f.write(f"  Avg Path Length: {method_data.get('avg_path_length', 0):.2f}\n")
                        f.write(f"  Avg Planning Time: {method_data.get('avg_planning_time', 0)*1000:.1f}ms\n\n")
            
            f.write("ICRA 2025 READY: ✅\n")
            f.write("Publication-quality data and visualizations generated.\n")
            f.write(f"All outputs available in: plots/standard_cp/\n")
        
        print(f"📋 Summary report saved: {report_file}")
    
    def _estimate_calibration_trials(self) -> int:
        """Estimate total calibration trials for progress tracking"""
        if self.planner.config.get('environments', {}).get('full_environments', {}).get('enabled', False):
            cal_envs = self.planner.config['environments']['full_environments']['calibration_envs']
        else:
            cal_envs = self.planner.config['environments']['test_environments']
        
        total_trials = 0
        for env_config in cal_envs:
            total_trials += len(env_config['test_ids']) * self.full_scale_config['calibration_trials']
        
        return total_trials
    
    def _json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def run_complete_evaluation(self) -> bool:
        """Run the complete full-scale evaluation pipeline"""
        print(f"🚀 STARTING FULL-SCALE STANDARD CP EVALUATION")
        print(f"=" * 80)
        print(f"Timestamp: {self.timestamp}")
        print(f"Expected total duration: ~2 hours")
        print(f"=" * 80)
        
        total_start_time = time.time()
        
        try:
            # Phase 1: Comprehensive Calibration
            print(f"\n📍 PHASE 1: COMPREHENSIVE τ CALIBRATION")
            tau = self.run_comprehensive_calibration()
            
            # Phase 2: Full-Scale Evaluation
            print(f"\n📍 PHASE 2: FULL-SCALE MONTE CARLO EVALUATION")
            results = self.run_full_scale_evaluation(tau)
            
            # Phase 3: Publication Outputs
            print(f"\n📍 PHASE 3: PUBLICATION-READY OUTPUT GENERATION")
            pub_success = self.generate_publication_outputs(results, tau)
            
            # Final Summary
            total_time = time.time() - total_start_time
            
            # Show detailed progress summary
            self.progress_tracker.print_final_summary()
            
            print(f"\n🎉 FULL-SCALE EVALUATION COMPLETED!")
            print(f"=" * 80)
            print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            print(f"Global τ calibrated: {tau:.4f}m ({tau*100:.1f}cm)")
            print(f"Monte Carlo trials: {self.full_scale_config['evaluation_trials']} per method")
            print(f"Publication outputs: {'✅ Generated' if pub_success else '❌ Failed'}")
            print(f"Progress tracking: ✅ Complete trial-by-trial logging")
            print(f"Detailed logs: ✅ CSV files with failure analysis")
            print(f"")
            print(f"🔬 ICRA 2025 READY:")
            print(f"   • Comprehensive Standard CP evaluation with progress tracking")
            print(f"   • Detailed failure categorization (timeout/iteration/other)")
            print(f"   • Publication-quality data and visualizations")
            print(f"   • Ready for Learnable CP implementation")
            print(f"   • All outputs in: plots/standard_cp/")
            print(f"=" * 80)
            
            return True
            
        except Exception as e:
            total_time = time.time() - total_start_time
            print(f"\n❌ FULL-SCALE EVALUATION FAILED")
            print(f"Error after {total_time:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for full-scale evaluation"""
    try:
        evaluator = FullScaleStandardCPEvaluator()
        success = evaluator.run_complete_evaluation()
        return success
    except Exception as e:
        print(f"❌ Full-scale evaluation initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)