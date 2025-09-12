#!/usr/bin/env python3
"""
Standard CP Visualization
Reuses existing visualization infrastructure for Standard CP results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import seaborn as sns

# Import existing visualization infrastructure
sys.path.append('.')
from visualization import MRPBVisualizer

class StandardCPVisualizer:
    """
    Visualization for Standard CP results
    Reuses existing MRPBVisualizer infrastructure
    """
    
    def __init__(self, base_dir="plots/standard_cp"):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.plots_dir = os.path.join(base_dir, "plots")
        self.calibration_dir = os.path.join(base_dir, "calibration")
        self.evaluation_dir = os.path.join(base_dir, "evaluation")
        
        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def find_latest_files(self):
        """Find the most recent data files"""
        files = {}
        
        # Look for latest files in each category
        for category, directory in [
            ('nonconformity', self.results_dir),
            ('calibration_summary', self.calibration_dir),
            ('tau_analysis', self.calibration_dir),
            ('method_comparison', self.evaluation_dir),
            ('planning_times', self.results_dir),
            ('path_lengths', self.results_dir)
        ]:
            pattern = f"{category}_"
            matching_files = [f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith('.csv')]
            
            if matching_files:
                # Get most recent file
                latest_file = max(matching_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
                files[category] = os.path.join(directory, latest_file)
                print(f"📊 Found {category}: {latest_file}")
        
        return files
    
    def create_calibration_plots(self, files):
        """Create calibration analysis plots"""
        print("\n🎨 Creating calibration plots...")
        
        # 1. Nonconformity scores distribution
        if 'nonconformity' in files:
            df = pd.read_csv(files['nonconformity'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Standard CP Calibration Analysis', fontsize=16, fontweight='bold')
            
            # Score distribution by environment
            axes[0, 0].hist([df[df['environment'] == env]['nonconformity_score'].values 
                           for env in df['environment'].unique()], 
                          bins=20, alpha=0.7, label=df['environment'].unique())
            axes[0, 0].set_title('Nonconformity Score Distribution by Environment')
            axes[0, 0].set_xlabel('Nonconformity Score (m)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Score vs noise level
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                axes[0, 1].scatter(env_data['noise_level'], env_data['nonconformity_score'], 
                                 alpha=0.6, label=env, s=30)
            axes[0, 1].set_title('Nonconformity Score vs Noise Level')
            axes[0, 1].set_xlabel('Noise Level')
            axes[0, 1].set_ylabel('Nonconformity Score (m)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Planning success rate by environment
            success_rates = df.groupby('environment')['path_found'].mean()
            axes[1, 0].bar(success_rates.index, success_rates.values, color='skyblue', alpha=0.8)
            axes[1, 0].set_title('Planning Success Rate by Environment')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Path length distribution for successful paths
            successful_paths = df[df['path_found'] == True]
            if len(successful_paths) > 0:
                axes[1, 1].boxplot([successful_paths[successful_paths['environment'] == env]['path_length'].values 
                                  for env in successful_paths['environment'].unique()], 
                                 labels=successful_paths['environment'].unique())
                axes[1, 1].set_title('Path Length Distribution (Successful Paths)')
                axes[1, 1].set_ylabel('Path Length (waypoints)')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calibration_plot = os.path.join(self.plots_dir, f"calibration_analysis_{timestamp}.png")
            plt.savefig(calibration_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Calibration plot saved: {calibration_plot}")
    
    def create_tau_analysis_plot(self, files):
        """Create tau analysis visualization"""
        if 'tau_analysis' in files:
            print("\n🎨 Creating tau analysis plot...")
            
            df = pd.read_csv(files['tau_analysis'])
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Tau (τ) Analysis - Global Safety Margin', fontsize=16, fontweight='bold')
            
            # Tau value with confidence intervals
            tau_val = df['tau_value'].iloc[0]
            confidence = df['confidence_level'].iloc[0] * 100
            
            axes[0].axhline(y=tau_val, color='red', linewidth=3, label=f'Global τ = {tau_val:.3f}m')
            axes[0].axhline(y=df['score_mean'].iloc[0], color='blue', linewidth=2, linestyle='--', 
                          label=f'Mean score = {df["score_mean"].iloc[0]:.3f}m')
            axes[0].axhline(y=df['score_90th_percentile'].iloc[0], color='orange', linewidth=2, linestyle=':', 
                          label=f'90th percentile = {df["score_90th_percentile"].iloc[0]:.3f}m')
            
            axes[0].set_title(f'Tau Value ({confidence:.0f}% Coverage)')
            axes[0].set_ylabel('Safety Margin (m)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, max(tau_val * 1.2, 0.3))
            
            # Score distribution with tau
            axes[1].hist([0, df['score_min'].iloc[0], df['score_mean'].iloc[0], 
                         df['score_90th_percentile'].iloc[0], df['score_max'].iloc[0]], 
                        bins=20, alpha=0.7, color='lightblue', edgecolor='blue')
            axes[1].axvline(x=tau_val, color='red', linewidth=3, label=f'τ = {tau_val:.3f}m')
            axes[1].set_title('Score Distribution with Tau Threshold')
            axes[1].set_xlabel('Nonconformity Score (m)')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tau_plot = os.path.join(self.plots_dir, f"tau_analysis_{timestamp}.png")
            plt.savefig(tau_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Tau analysis plot saved: {tau_plot}")
    
    def create_method_comparison_plot(self, files):
        """Create method comparison visualization"""
        if 'method_comparison' in files:
            print("\n🎨 Creating method comparison plot...")
            
            df = pd.read_csv(files['method_comparison'])
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Naive vs Standard CP Comparison', fontsize=16, fontweight='bold')
            
            methods = df['method'].unique()
            colors = ['skyblue', 'lightcoral']
            
            # Success rates
            success_rates = [df[df['method'] == method]['success_rate'].iloc[0] * 100 for method in methods]
            bars1 = axes[0, 0].bar(methods, success_rates, color=colors, alpha=0.8)
            axes[0, 0].set_title('Planning Success Rate')
            axes[0, 0].set_ylabel('Success Rate (%)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, value in zip(bars1, success_rates):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                              f'{value:.1f}%', ha='center', fontweight='bold')
            
            # Collision rates  
            collision_rates = [df[df['method'] == method]['collision_rate'].iloc[0] * 100 for method in methods]
            bars2 = axes[0, 1].bar(methods, collision_rates, color=colors, alpha=0.8)
            axes[0, 1].set_title('Collision Rate on True Environment')
            axes[0, 1].set_ylabel('Collision Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars2, collision_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                              f'{value:.1f}%', ha='center', fontweight='bold')
            
            # Planning times
            planning_times = [df[df['method'] == method]['avg_planning_time'].iloc[0] * 1000 for method in methods]
            bars3 = axes[1, 0].bar(methods, planning_times, color=colors, alpha=0.8)
            axes[1, 0].set_title('Average Planning Time')
            axes[1, 0].set_ylabel('Planning Time (ms)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars3, planning_times):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                              f'{value:.1f}ms', ha='center', fontweight='bold')
            
            # Path lengths
            path_lengths = [df[df['method'] == method]['avg_path_length'].iloc[0] for method in methods]
            bars4 = axes[1, 1].bar(methods, path_lengths, color=colors, alpha=0.8)
            axes[1, 1].set_title('Average Path Length')
            axes[1, 1].set_ylabel('Path Length (waypoints)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels and percentage increase
            for i, (bar, value) in enumerate(zip(bars4, path_lengths)):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                              f'{value:.1f}', ha='center', fontweight='bold')
            
            # Add percentage increase annotation
            if len(path_lengths) == 2 and path_lengths[0] > 0:
                increase = ((path_lengths[1] - path_lengths[0]) / path_lengths[0]) * 100
                axes[1, 1].text(0.5, max(path_lengths) * 0.9, 
                              f'+{increase:.1f}% safety overhead', 
                              ha='center', va='center',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_plot = os.path.join(self.plots_dir, f"method_comparison_{timestamp}.png")
            plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Method comparison plot saved: {comparison_plot}")
    
    def create_performance_analysis_plot(self, files):
        """Create detailed performance analysis"""
        if 'planning_times' in files and 'path_lengths' in files:
            print("\n🎨 Creating performance analysis plot...")
            
            times_df = pd.read_csv(files['planning_times'])
            lengths_df = pd.read_csv(files['path_lengths'])
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Standard CP Performance Analysis', fontsize=16, fontweight='bold')
            
            # Planning time distributions
            for method in times_df['method'].unique():
                method_data = times_df[times_df['method'] == method]['planning_time'] * 1000
                axes[0, 0].hist(method_data, bins=15, alpha=0.7, label=method, density=True)
            
            axes[0, 0].set_title('Planning Time Distribution')
            axes[0, 0].set_xlabel('Planning Time (ms)')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Path length distributions
            for method in lengths_df['method'].unique():
                method_data = lengths_df[lengths_df['method'] == method]['path_length']
                axes[0, 1].hist(method_data, bins=15, alpha=0.7, label=method, density=True)
            
            axes[0, 1].set_title('Path Length Distribution')
            axes[0, 1].set_xlabel('Path Length (waypoints)')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Planning time box plots
            time_data = [times_df[times_df['method'] == method]['planning_time'].values * 1000 
                        for method in times_df['method'].unique()]
            axes[1, 0].boxplot(time_data, labels=times_df['method'].unique())
            axes[1, 0].set_title('Planning Time Comparison (Box Plot)')
            axes[1, 0].set_ylabel('Planning Time (ms)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Path length box plots
            length_data = [lengths_df[lengths_df['method'] == method]['path_length'].values 
                          for method in lengths_df['method'].unique()]
            axes[1, 1].boxplot(length_data, labels=lengths_df['method'].unique())
            axes[1, 1].set_title('Path Length Comparison (Box Plot)')
            axes[1, 1].set_ylabel('Path Length (waypoints)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            performance_plot = os.path.join(self.plots_dir, f"performance_analysis_{timestamp}.png")
            plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Performance analysis plot saved: {performance_plot}")
    
    def create_summary_dashboard(self, files):
        """Create comprehensive summary dashboard"""
        print("\n🎨 Creating summary dashboard...")
        
        # Load key data
        tau_df = pd.read_csv(files['tau_analysis']) if 'tau_analysis' in files else None
        comparison_df = pd.read_csv(files['method_comparison']) if 'method_comparison' in files else None
        
        if tau_df is not None and comparison_df is not None:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Standard CP Complete Analysis Dashboard', fontsize=20, fontweight='bold')
            
            # Create subplot layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # Tau summary (top left)
            ax1 = fig.add_subplot(gs[0, :2])
            tau_val = tau_df['tau_value'].iloc[0]
            confidence = tau_df['confidence_level'].iloc[0] * 100
            trials = tau_df['total_trials'].iloc[0]
            
            ax1.text(0.5, 0.7, f'Global Safety Margin', fontsize=18, fontweight='bold', 
                    ha='center', transform=ax1.transAxes)
            ax1.text(0.5, 0.5, f'τ = {tau_val:.3f}m', fontsize=24, fontweight='bold', 
                    ha='center', transform=ax1.transAxes, color='red')
            ax1.text(0.5, 0.3, f'{confidence:.0f}% Coverage | {trials} Trials', fontsize=14, 
                    ha='center', transform=ax1.transAxes)
            ax1.axis('off')
            
            # Key metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            naive_data = comparison_df[comparison_df['method'] == 'naive'].iloc[0]
            cp_data = comparison_df[comparison_df['method'] == 'standard_cp'].iloc[0]
            
            metrics_text = f"""
Standard CP vs Naive Results:
────────────────────────────
Success Rate: {cp_data['success_rate']*100:.1f}% vs {naive_data['success_rate']*100:.1f}%
Collision Rate: {cp_data['collision_rate']*100:.1f}% vs {naive_data['collision_rate']*100:.1f}%
Path Length: {cp_data['avg_path_length']:.1f} vs {naive_data['avg_path_length']:.1f} (+{((cp_data['avg_path_length']/naive_data['avg_path_length']-1)*100):.1f}%)
Planning Time: {cp_data['avg_planning_time']*1000:.1f}ms vs {naive_data['avg_planning_time']*1000:.1f}ms
"""
            ax2.text(0.05, 0.95, metrics_text, fontsize=12, fontfamily='monospace',
                    ha='left', va='top', transform=ax2.transAxes)
            ax2.axis('off')
            
            # Method comparison chart (bottom)
            ax3 = fig.add_subplot(gs[1:, :])
            
            methods = comparison_df['method'].unique()
            x = np.arange(len(methods))
            width = 0.2
            
            # Multiple metrics
            success_rates = [comparison_df[comparison_df['method'] == method]['success_rate'].iloc[0] * 100 for method in methods]
            collision_rates = [comparison_df[comparison_df['method'] == method]['collision_rate'].iloc[0] * 100 for method in methods]
            path_lengths = [comparison_df[comparison_df['method'] == method]['avg_path_length'].iloc[0] for method in methods]
            planning_times = [comparison_df[comparison_df['method'] == method]['avg_planning_time'].iloc[0] * 1000 for method in methods]
            
            # Normalize metrics for comparison
            path_lengths_norm = [(l / max(path_lengths)) * 100 for l in path_lengths]
            planning_times_norm = [(t / max(planning_times)) * 100 if max(planning_times) > 0 else 0 for t in planning_times]
            
            ax3.bar(x - 1.5*width, success_rates, width, label='Success Rate (%)', alpha=0.8)
            ax3.bar(x - 0.5*width, [100 - cr for cr in collision_rates], width, label='Safety Rate (%)', alpha=0.8)
            ax3.bar(x + 0.5*width, path_lengths_norm, width, label='Path Length (normalized)', alpha=0.8)
            ax3.bar(x + 1.5*width, planning_times_norm, width, label='Planning Time (normalized)', alpha=0.8)
            
            ax3.set_title('Comprehensive Method Comparison', fontsize=16, fontweight='bold')
            ax3.set_ylabel('Metric Value')
            ax3.set_xticks(x)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in methods])
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_plot = os.path.join(self.plots_dir, f"summary_dashboard_{timestamp}.png")
            plt.savefig(dashboard_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Summary dashboard saved: {dashboard_plot}")
    
    def generate_all_visualizations(self):
        """Generate all Standard CP visualizations"""
        print("🎨 STANDARD CP VISUALIZATION GENERATOR")
        print("="*60)
        
        # Find latest data files
        files = self.find_latest_files()
        
        if not files:
            print("❌ No data files found. Run Standard CP evaluation first.")
            return False
        
        # Generate all visualizations
        self.create_calibration_plots(files)
        self.create_tau_analysis_plot(files)
        self.create_method_comparison_plot(files)
        self.create_performance_analysis_plot(files)
        self.create_summary_dashboard(files)
        
        print(f"\n🎉 All visualizations completed!")
        print(f"📁 Plots saved in: {self.plots_dir}")
        
        return True


def main():
    """Generate Standard CP visualizations"""
    visualizer = StandardCPVisualizer()
    success = visualizer.generate_all_visualizations()
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)