import json
import datetime
import csv
import os

class MemorySystem:
    def __init__(self):
        self.progress_file = 'icra_implementation/PROGRESS.md'
        self.metrics_file = 'icra_implementation/results/metrics.csv'
        self.state_file = 'icra_implementation/checkpoints/current_state.json'
        self.log_file = 'icra_implementation/logs/detailed_log.txt'
        
    def log_progress(self, task, status, details):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.progress_file, 'a') as f:
            f.write(f"\n## [{timestamp}] {task}\n")
            f.write(f"**Status:** {status}\n")
            f.write(f"**Details:** {details}\n")
            f.write("-" * 50 + "\n")
            
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {task} - {status}: {details}\n")
        
        print(f"[{status}] {task}: {details}")
            
    def save_metrics(self, metrics_dict):
        # Add timestamp
        metrics_dict['timestamp'] = datetime.datetime.now().isoformat()
        
        # Check if file exists and has header
        file_exists = os.path.exists(self.metrics_file) and os.path.getsize(self.metrics_file) > 0
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_dict)
            
    def checkpoint(self, state_dict):
        state_dict['timestamp'] = datetime.datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=2)
            
    def load_checkpoint(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None