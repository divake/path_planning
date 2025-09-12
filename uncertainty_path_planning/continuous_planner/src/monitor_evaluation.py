#!/usr/bin/env python3
"""
Evaluation Progress Monitor
Monitor the progress of the full-scale Standard CP evaluation
"""

import time
import json
from pathlib import Path
from datetime import datetime
import glob

def monitor_evaluation_progress():
    """Monitor the progress of the full-scale evaluation"""
    print("🔍 MONITORING STANDARD CP FULL-SCALE EVALUATION")
    print("=" * 60)
    
    base_dir = Path("plots/standard_cp")
    
    while True:
        print(f"\n⏰ Status Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 40)
        
        # Check for log files
        log_files = list(base_dir.glob("logs/*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
            print(f"📝 Latest log: {latest_log.name}")
            
            # Read last few lines of log
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("   Recent activity:")
                        for line in lines[-3:]:
                            print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Could not read log: {e}")
        
        # Check for intermediate results
        result_files = list(base_dir.glob("results/*.json"))
        if result_files:
            print(f"📊 Results files found: {len(result_files)}")
            for rf in result_files[-2:]:  # Show latest 2
                print(f"   {rf.name}")
        
        # Check for CSV files
        csv_files = list(base_dir.glob("results/*.csv"))
        if csv_files:
            print(f"📈 CSV files found: {len(csv_files)}")
            
        # Check for plots
        plot_files = list(base_dir.glob("plots/*.png"))
        if plot_files:
            print(f"🎨 Plot files found: {len(plot_files)}")
        
        # Check if evaluation completed
        completion_files = list(base_dir.glob("results/full_evaluation_*.json"))
        if completion_files:
            print(f"🎉 EVALUATION COMPLETED!")
            latest_result = max(completion_files, key=lambda f: f.stat().st_mtime)
            
            try:
                with open(latest_result, 'r') as f:
                    data = json.load(f)
                    
                exp_info = data.get('experiment_info', {})
                print(f"   Timestamp: {exp_info.get('timestamp', 'unknown')}")
                print(f"   Total trials: {exp_info.get('total_trials', 'unknown')}")
                print(f"   Global τ: {exp_info.get('global_tau', 'unknown')}")
                
                results = data.get('results', {})
                if 'standard_cp' in results:
                    cp_data = results['standard_cp']
                    print(f"   Standard CP Success: {cp_data.get('success_rate', 0)*100:.1f}%")
                    print(f"   Standard CP Collisions: {cp_data.get('collision_rate', 0)*100:.1f}%")
                
            except Exception as e:
                print(f"   Could not parse results: {e}")
            
            break
        
        print(f"💭 Evaluation in progress... (will check again in 60s)")
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor_evaluation_progress()