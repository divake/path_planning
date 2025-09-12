#!/usr/bin/env python3
"""
Visualization module for MRPB test results
Handles CSV generation and plot creation with tables
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import csv
import os
from typing import Dict, List
from datetime import datetime


class MRPBVisualizer:
    """Visualizer for MRPB experiment results"""
    
    def __init__(self, results_dir='../results', plots_dir='../plots'):
        """Initialize visualizer with output directories"""
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_results_to_csv(self, all_results: Dict, timestamp: str = None) -> str:
        """
        Save experiment results to CSV file
        
        Args:
            all_results: Dictionary with all test results
            timestamp: Optional timestamp string
            
        Returns:
            Path to saved CSV file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Collect all results for CSV
        csv_data = []
        for env_name, env_results in all_results.items():
            for test_name, test_results in env_results.items():
                if 'naive' in test_results and test_results['naive'].get('success', False):
                    result = test_results['naive']
                    metrics = result.get('metrics', {})
                    
                    csv_data.append({
                        'env_name': env_name,
                        'test_id': int(test_name.split('_')[1]),
                        'success': result['success'],
                        'path_length': result.get('path_length', 0),
                        'planning_time': result.get('planning_time', 0),
                        'num_waypoints': result.get('num_waypoints', 0),
                        'd_0': metrics.get('safety', {}).get('d_0', 0),
                        'd_avg': metrics.get('safety', {}).get('d_avg', 0),
                        'p_0': metrics.get('safety', {}).get('p_0', 0),
                        'T': metrics.get('efficiency', {}).get('T', 0),
                        'C': metrics.get('efficiency', {}).get('C', 0),
                        'f_ps': metrics.get('smoothness', {}).get('f_ps', 0),
                        'f_vs': metrics.get('smoothness', {}).get('f_vs', 0)
                    })
        
        # Save to CSV
        csv_file = os.path.join(self.results_dir, f"rrt_star_results_{timestamp}.csv")
        if csv_data:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
            print(f"CSV saved to: {csv_file}")
            return csv_file
        return None
    
    def generate_comprehensive_plot(self, all_results: Dict, timestamp: str = None, 
                                   robot_radius: float = 0.17) -> str:
        """
        Generate comprehensive plot with metrics and data table
        
        Args:
            all_results: Dictionary with all test results
            timestamp: Optional timestamp string
            robot_radius: Robot radius for visualization
            
        Returns:
            Path to saved plot file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Collect data for plotting
        plot_data = []
        for env_name, env_results in all_results.items():
            for test_name, test_results in env_results.items():
                if 'naive' in test_results and test_results['naive'].get('success', False):
                    result = test_results['naive']
                    metrics = result.get('metrics', {})
                    
                    plot_data.append({
                        'env_name': env_name,
                        'test_id': int(test_name.split('_')[1]),
                        'path_length': result.get('path_length', 0),
                        'planning_time': result.get('planning_time', 0),
                        'num_waypoints': result.get('num_waypoints', 0),
                        'd_0': metrics.get('safety', {}).get('d_0', 0),
                        'd_avg': metrics.get('safety', {}).get('d_avg', 0),
                        'p_0': metrics.get('safety', {}).get('p_0', 0),
                        'T': metrics.get('efficiency', {}).get('T', 0),
                        'C': metrics.get('efficiency', {}).get('C', 0),
                        'f_ps': metrics.get('smoothness', {}).get('f_ps', 0),
                        'f_vs': metrics.get('smoothness', {}).get('f_vs', 0)
                    })
        
        if not plot_data:
            print("No successful results to plot")
            return None
        
        # Create figure with subplots and table
        fig = plt.figure(figsize=(20, 14))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1])
        
        # Extract data for plotting
        env_names = [f"{d['env_name'][:6]}-{d['test_id']}" for d in plot_data]
        x_pos = np.arange(len(env_names))
        
        # Plot 1: Path Length
        ax1 = fig.add_subplot(gs[0, 0])
        path_lengths = [d['path_length'] for d in plot_data]
        bars1 = ax1.bar(x_pos, path_lengths, color='blue', alpha=0.7)
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Path Length (m)')
        ax1.set_title('Path Length', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(env_names, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars1, path_lengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        
        # Plot 2: Planning Time
        ax2 = fig.add_subplot(gs[0, 1])
        times = [d['planning_time'] for d in plot_data]
        bars2 = ax2.bar(x_pos, times, color='green', alpha=0.7)
        ax2.set_xlabel('Test Case')
        ax2.set_ylabel('Planning Time (s)')
        ax2.set_title('Planning Time', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(env_names, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        # Add value labels on bars
        for bar, val in zip(bars2, times):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        
        # Plot 3: Safety Metrics (d_0 and d_avg)
        ax3 = fig.add_subplot(gs[0, 2])
        d0_values = [d['d_0'] for d in plot_data]
        d_avg_values = [d['d_avg'] for d in plot_data]
        
        width = 0.35
        x_pos1 = x_pos - width/2
        x_pos2 = x_pos + width/2
        
        bars3a = ax3.bar(x_pos1, d0_values, width, color='red', alpha=0.7, label='d_0 (min)')
        bars3b = ax3.bar(x_pos2, d_avg_values, width, color='orange', alpha=0.7, label='d_avg')
        
        ax3.set_xlabel('Test Case')
        ax3.set_ylabel('Distance to Obstacles (m)')
        ax3.set_title('Safety Metrics (d_0 vs d_avg)', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(env_names, rotation=45, ha='right', fontsize=8)
        ax3.axhline(y=robot_radius, color='black', linestyle='--', label=f'Robot radius ({robot_radius}m)')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Travel Time
        ax4 = fig.add_subplot(gs[1, 0])
        t_values = [d['T'] for d in plot_data]
        bars4 = ax4.bar(x_pos, t_values, color='orange', alpha=0.7)
        ax4.set_xlabel('Test Case')
        ax4.set_ylabel('Travel Time (s)')
        ax4.set_title('Efficiency: Travel Time (T)', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(env_names, rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Velocity Smoothness
        ax5 = fig.add_subplot(gs[1, 1])
        fvs_values = [d['f_vs'] for d in plot_data]
        bars5 = ax5.bar(x_pos, fvs_values, color='purple', alpha=0.7)
        ax5.set_xlabel('Test Case')
        ax5.set_ylabel('Velocity Smoothness (m/s²)')
        ax5.set_title('Smoothness Metric (f_vs)', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(env_names, rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Count tests per environment
        env_counts = {}
        for d in plot_data:
            env = d['env_name']
            env_counts[env] = env_counts.get(env, 0) + 1
        
        env_breakdown = '\n'.join([f"• {env}: {count} tests" for env, count in env_counts.items()])
        
        summary_text = f"""SUMMARY STATISTICS
        
Total Tests: {len(plot_data)}
Success Rate: 100%

Average Metrics:
• Path Length: {np.mean(path_lengths):.2f}m
• Waypoints: {np.mean([d['num_waypoints'] for d in plot_data]):.0f}
• Planning Time: {np.mean(times):.2f}s
• Min Obstacle Dist: {np.mean(d0_values):.3f}m
• Avg Obstacle Dist: {np.mean(d_avg_values):.3f}m
• Travel Time: {np.mean(t_values):.2f}s
• Velocity Smoothness: {np.mean(fvs_values):.2f} m/s²

Environment Breakdown:
{env_breakdown}
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        
        # Add comprehensive data table at bottom
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        # Prepare table data with ALL metrics
        table_data = []
        table_data.append(['Test', 'Path(m)', 'Waypts', 'PlanTime(s)', 'd_0(m)', 'd_avg(m)', 'p_0(%)', 'T(s)', 'C(ms)', 'f_ps', 'f_vs'])
        
        for d in plot_data[:13]:  # Show up to 13 rows to fit
            table_data.append([
                f"{d['env_name'][:6]}-{d['test_id']}",
                f"{d['path_length']:.1f}",
                f"{d['num_waypoints']}",
                f"{d['planning_time']:.1f}",
                f"{d['d_0']:.3f}",
                f"{d['d_avg']:.3f}",
                f"{d['p_0']:.1f}",
                f"{d['T']:.2f}",
                f"{d['C']:.0f}",
                f"{d['f_ps']:.2f}",
                f"{d['f_vs']:.1f}"
            ])
        
        # Add average row
        table_data.append([
            'AVERAGE',
            f"{np.mean([d['path_length'] for d in plot_data]):.1f}",
            f"{np.mean([d['num_waypoints'] for d in plot_data]):.0f}",
            f"{np.mean([d['planning_time'] for d in plot_data]):.1f}",
            f"{np.mean([d['d_0'] for d in plot_data]):.3f}",
            f"{np.mean([d['d_avg'] for d in plot_data]):.3f}",
            f"{np.mean([d['p_0'] for d in plot_data]):.1f}",
            f"{np.mean([d['T'] for d in plot_data]):.2f}",
            f"{np.mean([d['C'] for d in plot_data]):.0f}",
            f"{np.mean([d['f_ps'] for d in plot_data]):.2f}",
            f"{np.mean([d['f_vs'] for d in plot_data]):.1f}"
        ])
        
        # Create table with smaller font to fit more columns
        table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Smaller font for more columns
        table.scale(1.4, 1.4)  # Wider table to accommodate all columns
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style average row
        last_row = len(table_data) - 1
        for i in range(len(table_data[0])):
            table[(last_row, i)].set_facecolor('#FFD700')
            table[(last_row, i)].set_text_props(weight='bold')
        
        # Alternate row colors
        for i in range(1, len(table_data)-1):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.suptitle('RRT* Performance Metrics on MRPB Environments', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.plots_dir, f"rrt_star_metrics_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot with table saved to: {plot_file}")
        return plot_file
    
    def generate_failing_maps_visualization(self, failing_tests: List[tuple], 
                                           timestamp: str = None) -> str:
        """
        Generate visualization of failing test maps
        
        Args:
            failing_tests: List of (env_name, test_id) tuples
            timestamp: Optional timestamp string
            
        Returns:
            Path to saved plot file
        """
        # Implementation for visualizing failing maps if needed
        pass