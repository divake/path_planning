#!/usr/bin/env python3
"""
Standard CP Progress Tracker
Real-time progress visualization and detailed failure logging
"""

import time
import csv
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm not available. Install with: pip install tqdm")

class StandardCPProgressTracker:
    """Progress tracker with visual progress bars and detailed logging"""
    
    def __init__(self, base_dir: str = "plots/standard_cp"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Progress tracking
        self.calibration_progress = None
        self.evaluation_progress = None
        
        # Detailed logging
        self.detailed_logs = []
        self.failure_logs = []
        self.timing_logs = []
        
        # Statistics
        self.stats = {
            'calibration': {
                'total_trials': 0,
                'successful_trials': 0,
                'failed_trials': 0,
                'timeout_failures': 0,
                'iteration_failures': 0,
                'other_failures': 0
            },
            'evaluation': {
                'total_trials': 0,
                'successful_trials': 0,
                'failed_trials': 0,
                'timeout_failures': 0,
                'iteration_failures': 0,
                'other_failures': 0
            }
        }
        
        # Setup logging directory
        self.logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"📊 PROGRESS TRACKER INITIALIZED")
        print(f"   Detailed logging enabled")
        print(f"   Progress bars: {'✅ Available' if TQDM_AVAILABLE else '❌ Not available'}")
        print(f"   Logs directory: {self.logs_dir}")
    
    def init_calibration_progress(self, total_trials: int):
        """Initialize calibration progress bar"""
        self.stats['calibration']['total_trials'] = total_trials
        
        if TQDM_AVAILABLE:
            self.calibration_progress = tqdm(
                total=total_trials,
                desc="🔧 Calibration",
                unit="trial",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}",
                file=sys.stdout
            )
            self.calibration_progress.set_postfix_str("0/0 (0.0%)")
        else:
            print(f"🔧 CALIBRATION PROGRESS: 0/{total_trials} (0.0%)")
        
        # Initialize CSV logging
        self.calibration_csv_file = os.path.join(self.logs_dir, f"calibration_detailed_{self.timestamp}.csv")
        with open(self.calibration_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_id', 'environment', 'test_id', 'method', 'success', 'failure_reason',
                'planning_time', 'iterations_used', 'path_length', 'timestamp'
            ])
    
    def update_calibration_progress(self, trial_info: Dict[str, Any]):
        """Update calibration progress"""
        success = trial_info.get('success', False)
        failure_reason = trial_info.get('failure_reason', '')
        
        if success:
            self.stats['calibration']['successful_trials'] += 1
        else:
            self.stats['calibration']['failed_trials'] += 1
            if 'timeout' in failure_reason.lower():
                self.stats['calibration']['timeout_failures'] += 1
            elif 'iteration' in failure_reason.lower():
                self.stats['calibration']['iteration_failures'] += 1
            else:
                self.stats['calibration']['other_failures'] += 1
        
        # Log detailed information
        self.log_trial_details(trial_info, 'calibration')
        
        # Update progress bar
        if TQDM_AVAILABLE and self.calibration_progress:
            self.calibration_progress.update(1)
            success_rate = (self.stats['calibration']['successful_trials'] / 
                          max(1, self.stats['calibration']['successful_trials'] + self.stats['calibration']['failed_trials'])) * 100
            self.calibration_progress.set_postfix_str(
                f"{self.stats['calibration']['successful_trials']}/{self.stats['calibration']['successful_trials'] + self.stats['calibration']['failed_trials']} ({success_rate:.1f}%)"
            )
        else:
            completed = self.stats['calibration']['successful_trials'] + self.stats['calibration']['failed_trials']
            total = self.stats['calibration']['total_trials']
            success_rate = (self.stats['calibration']['successful_trials'] / max(1, completed)) * 100
            print(f"\r🔧 CALIBRATION PROGRESS: {completed}/{total} ({completed/total*100:.1f}%) Success: {success_rate:.1f}%", end='', flush=True)
    
    def finish_calibration_progress(self):
        """Finish calibration progress and show summary"""
        if TQDM_AVAILABLE and self.calibration_progress:
            self.calibration_progress.close()
        else:
            print()  # New line after progress updates
        
        self.print_calibration_summary()
    
    def init_evaluation_progress(self, total_trials: int):
        """Initialize evaluation progress bar"""
        self.stats['evaluation']['total_trials'] = total_trials
        
        if TQDM_AVAILABLE:
            self.evaluation_progress = tqdm(
                total=total_trials,
                desc="🧪 Evaluation",
                unit="trial",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}",
                file=sys.stdout
            )
            self.evaluation_progress.set_postfix_str("0/0 (0.0%)")
        else:
            print(f"🧪 EVALUATION PROGRESS: 0/{total_trials} (0.0%)")
        
        # Initialize CSV logging
        self.evaluation_csv_file = os.path.join(self.logs_dir, f"evaluation_detailed_{self.timestamp}.csv")
        with open(self.evaluation_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_id', 'environment', 'test_id', 'method', 'success', 'failure_reason',
                'planning_time', 'iterations_used', 'path_length', 'collision', 'timestamp'
            ])
    
    def update_evaluation_progress(self, trial_info: Dict[str, Any]):
        """Update evaluation progress"""
        success = trial_info.get('success', False)
        failure_reason = trial_info.get('failure_reason', '')
        
        if success:
            self.stats['evaluation']['successful_trials'] += 1
        else:
            self.stats['evaluation']['failed_trials'] += 1
            if 'timeout' in failure_reason.lower():
                self.stats['evaluation']['timeout_failures'] += 1
            elif 'iteration' in failure_reason.lower():
                self.stats['evaluation']['iteration_failures'] += 1
            else:
                self.stats['evaluation']['other_failures'] += 1
        
        # Log detailed information
        self.log_trial_details(trial_info, 'evaluation')
        
        # Update progress bar
        if TQDM_AVAILABLE and self.evaluation_progress:
            self.evaluation_progress.update(1)
            success_rate = (self.stats['evaluation']['successful_trials'] / 
                          max(1, self.stats['evaluation']['successful_trials'] + self.stats['evaluation']['failed_trials'])) * 100
            self.evaluation_progress.set_postfix_str(
                f"{self.stats['evaluation']['successful_trials']}/{self.stats['evaluation']['successful_trials'] + self.stats['evaluation']['failed_trials']} ({success_rate:.1f}%)"
            )
        else:
            completed = self.stats['evaluation']['successful_trials'] + self.stats['evaluation']['failed_trials']
            total = self.stats['evaluation']['total_trials']
            success_rate = (self.stats['evaluation']['successful_trials'] / max(1, completed)) * 100
            print(f"\r🧪 EVALUATION PROGRESS: {completed}/{total} ({completed/total*100:.1f}%) Success: {success_rate:.1f}%", end='', flush=True)
    
    def finish_evaluation_progress(self):
        """Finish evaluation progress and show summary"""
        if TQDM_AVAILABLE and self.evaluation_progress:
            self.evaluation_progress.close()
        else:
            print()  # New line after progress updates
        
        self.print_evaluation_summary()
    
    def log_trial_details(self, trial_info: Dict[str, Any], phase: str):
        """Log detailed trial information to CSV"""
        csv_file = self.calibration_csv_file if phase == 'calibration' else self.evaluation_csv_file
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_info.get('trial_id', ''),
                trial_info.get('environment', ''),
                trial_info.get('test_id', ''),
                trial_info.get('method', ''),
                trial_info.get('success', False),
                trial_info.get('failure_reason', ''),
                trial_info.get('planning_time', 0.0),
                trial_info.get('iterations_used', 0),
                trial_info.get('path_length', 0.0),
                trial_info.get('collision', False) if phase == 'evaluation' else '',
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
    
    def print_calibration_summary(self):
        """Print calibration phase summary"""
        stats = self.stats['calibration']
        total_completed = stats['successful_trials'] + stats['failed_trials']
        success_rate = (stats['successful_trials'] / max(1, total_completed)) * 100
        
        print(f"\n📊 CALIBRATION PHASE SUMMARY:")
        print(f"   Total trials: {total_completed}/{stats['total_trials']}")
        print(f"   ✅ Successful: {stats['successful_trials']} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {stats['failed_trials']}")
        if stats['failed_trials'] > 0:
            print(f"      • Timeout failures: {stats['timeout_failures']}")
            print(f"      • Iteration failures: {stats['iteration_failures']}")  
            print(f"      • Other failures: {stats['other_failures']}")
        print(f"   📋 Detailed log: {os.path.basename(self.calibration_csv_file)}")
    
    def print_evaluation_summary(self):
        """Print evaluation phase summary"""
        stats = self.stats['evaluation']
        total_completed = stats['successful_trials'] + stats['failed_trials']
        success_rate = (stats['successful_trials'] / max(1, total_completed)) * 100
        
        print(f"\n📊 EVALUATION PHASE SUMMARY:")
        print(f"   Total trials: {total_completed}/{stats['total_trials']}")
        print(f"   ✅ Successful: {stats['successful_trials']} ({success_rate:.1f}%)")
        print(f"   ❌ Failed: {stats['failed_trials']}")
        if stats['failed_trials'] > 0:
            print(f"      • Timeout failures: {stats['timeout_failures']}")
            print(f"      • Iteration failures: {stats['iteration_failures']}")
            print(f"      • Other failures: {stats['other_failures']}")
        print(f"   📋 Detailed log: {os.path.basename(self.evaluation_csv_file)}")
    
    def print_final_summary(self):
        """Print final comprehensive summary"""
        print(f"\n🎯 FINAL PROGRESS SUMMARY")
        print(f"=" * 60)
        
        # Calibration summary
        cal_stats = self.stats['calibration']
        cal_completed = cal_stats['successful_trials'] + cal_stats['failed_trials']
        cal_success_rate = (cal_stats['successful_trials'] / max(1, cal_completed)) * 100
        
        print(f"📊 CALIBRATION RESULTS:")
        print(f"   Trials: {cal_completed}/{cal_stats['total_trials']}")
        print(f"   Success rate: {cal_success_rate:.1f}% ({cal_stats['successful_trials']}/{cal_completed})")
        print(f"   Failures: {cal_stats['timeout_failures']} timeout, {cal_stats['iteration_failures']} iterations, {cal_stats['other_failures']} other")
        
        # Evaluation summary
        eval_stats = self.stats['evaluation']
        eval_completed = eval_stats['successful_trials'] + eval_stats['failed_trials']
        eval_success_rate = (eval_stats['successful_trials'] / max(1, eval_completed)) * 100
        
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Trials: {eval_completed}/{eval_stats['total_trials']}")
        print(f"   Success rate: {eval_success_rate:.1f}% ({eval_stats['successful_trials']}/{eval_completed})")
        print(f"   Failures: {eval_stats['timeout_failures']} timeout, {eval_stats['iteration_failures']} iterations, {eval_stats['other_failures']} other")
        
        print(f"\n📁 DETAILED LOGS:")
        if hasattr(self, 'calibration_csv_file'):
            print(f"   Calibration: {self.calibration_csv_file}")
        if hasattr(self, 'evaluation_csv_file'):
            print(f"   Evaluation: {self.evaluation_csv_file}")
        
        print(f"=" * 60)
    
    def log_environment_timing(self, env_name: str, phase: str, duration: float):
        """Log timing information for environments"""
        timing_info = {
            'environment': env_name,
            'phase': phase,
            'duration': duration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.timing_logs.append(timing_info)
    
    def save_timing_summary(self):
        """Save timing summary to CSV"""
        timing_file = os.path.join(self.logs_dir, f"timing_summary_{self.timestamp}.csv")
        with open(timing_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['environment', 'phase', 'duration_seconds', 'timestamp'])
            for log in self.timing_logs:
                writer.writerow([log['environment'], log['phase'], log['duration'], log['timestamp']])
        print(f"📊 Timing summary saved: {os.path.basename(timing_file)}")


def install_tqdm():
    """Helper function to install tqdm if needed"""
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        print("✅ tqdm installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install tqdm: {e}")
        return False


if __name__ == "__main__":
    # Test the progress tracker
    tracker = StandardCPProgressTracker()
    
    print("Testing progress tracking...")
    
    # Test calibration progress
    tracker.init_calibration_progress(10)
    for i in range(10):
        trial_info = {
            'trial_id': i+1,
            'environment': 'office01add',
            'test_id': 1,
            'method': 'naive',
            'success': i % 3 != 0,  # Simulate some failures
            'failure_reason': 'timeout' if i % 5 == 0 else 'planning failed',
            'planning_time': 1.5 + i * 0.1,
            'iterations_used': 1000 if i % 3 != 0 else 50000,
            'path_length': 15.0 + i
        }
        tracker.update_calibration_progress(trial_info)
        time.sleep(0.1)
    
    tracker.finish_calibration_progress()
    tracker.print_final_summary()