#!/usr/bin/env python3
"""
Continuous Planner Visualization Module
All plotting and visualization functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional
import os


class ContinuousVisualizer:
    """
    Visualization utilities for continuous planning
    """
    
    def __init__(self, save_dir: str = "continuous_planner/results"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_environment(self, ax, obstacles: List, 
                        color: str = 'gray', alpha: float = 0.5,
                        label: str = None):
        """
        Plot obstacles in environment
        
        Args:
            ax: Matplotlib axis
            obstacles: List of (x, y, width, height) rectangles
            color: Obstacle color
            alpha: Transparency
            label: Legend label
        """
        for i, (x, y, w, h) in enumerate(obstacles):
            rect = patches.Rectangle((x, y), w, h,
                                    linewidth=1, edgecolor='black',
                                    facecolor=color, alpha=alpha,
                                    label=label if i == 0 else None)
            ax.add_patch(rect)
    
    def plot_path(self, ax, path: List[Tuple[float, float]],
                 color: str = 'blue', linewidth: float = 2,
                 label: str = None, style: str = '-'):
        """
        Plot path
        
        Args:
            ax: Matplotlib axis
            path: List of (x, y) points
            color: Path color
            linewidth: Line width
            label: Legend label
            style: Line style
        """
        if not path:
            return
        
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, style, color=color, 
               linewidth=linewidth, label=label, alpha=0.8)
    
    def plot_tree(self, ax, edges: List, color: str = 'lightgray',
                 alpha: float = 0.3):
        """
        Plot RRT* tree
        
        Args:
            ax: Matplotlib axis
            edges: List of (node1, node2) edges
            color: Edge color
            alpha: Transparency
        """
        for parent, child in edges:
            ax.plot([parent.x, child.x], [parent.y, child.y],
                   color=color, linewidth=0.5, alpha=alpha)
    
    def plot_comparison(self, results: Dict, title: str = "Method Comparison"):
        """
        Plot comparison between methods
        
        Args:
            results: Dictionary with method results
            title: Plot title
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        methods = list(results.keys())
        
        for idx, method in enumerate(methods[:6]):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.set_title(f'{method}', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 30)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            # Plot environment
            if 'perceived_obs' in results[method]:
                self.plot_environment(ax, results[method]['perceived_obs'],
                                    color='lightgray', alpha=0.3)
            
            if 'inflated_obs' in results[method]:
                self.plot_environment(ax, results[method]['inflated_obs'],
                                    color='orange', alpha=0.2)
            
            # Plot path
            if 'path' in results[method] and results[method]['path']:
                self.plot_path(ax, results[method]['path'],
                             color='blue' if 'naive' in method.lower() else 'green')
                
                # Check for collisions
                if 'collisions' in results[method]:
                    for col in results[method]['collisions']:
                        ax.plot(col[0], col[1], 'rx', markersize=10, markeredgewidth=2)
            
            # Add metrics
            if 'metrics' in results[method]:
                metrics = results[method]['metrics']
                text = f"Path Length: {metrics.get('path_length', 'N/A'):.1f}\n"
                text += f"Collisions: {metrics.get('num_collisions', 0)}"
                
                color = 'lightcoral' if metrics.get('num_collisions', 0) > 0 else 'lightgreen'
                ax.text(2, 28, text, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                       verticalalignment='top')
            
            # Mark start and goal
            ax.plot(5, 15, 'go', markersize=8)
            ax.plot(45, 15, 'ro', markersize=8)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        save_path = os.path.join(self.save_dir, 'method_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
        plt.close()
    
    def plot_tau_analysis(self, tau_curves: Dict, coverage_data: Dict):
        """
        Plot τ analysis and coverage curves
        
        Args:
            tau_curves: Dictionary of τ values for different confidence levels
            coverage_data: Coverage data for different methods
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: τ vs Confidence for different score types
        ax = axes[0, 0]
        ax.set_title('τ vs Confidence Level', fontweight='bold')
        
        for score_type, curve in tau_curves.items():
            confidences = list(curve.keys())
            taus = list(curve.values())
            ax.plot([c*100 for c in confidences], taus, 'o-', 
                   label=score_type, linewidth=2, markersize=6)
        
        ax.set_xlabel('Confidence Level (%)')
        ax.set_ylabel('τ (Safety Margin)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs τ
        ax = axes[0, 1]
        ax.set_title('Empirical Coverage vs τ', fontweight='bold')
        
        for method, data in coverage_data.items():
            if 'tau_values' in data and 'coverages' in data:
                ax.plot(data['tau_values'], data['coverages'], 'o-',
                       label=method, linewidth=2, markersize=6)
        
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% target')
        ax.set_xlabel('τ (Safety Margin)')
        ax.set_ylabel('Coverage (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Score distribution
        ax = axes[1, 0]
        ax.set_title('Nonconformity Score Distribution', fontweight='bold')
        
        if 'score_distributions' in coverage_data:
            for score_type, scores in coverage_data['score_distributions'].items():
                ax.hist(scores, bins=30, alpha=0.5, label=score_type, edgecolor='black')
        
        ax.set_xlabel('Nonconformity Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Plot 4: Method comparison table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create comparison table
        if 'comparison_table' in coverage_data:
            table_data = coverage_data['comparison_table']
            table = ax.table(cellText=table_data, loc='center',
                           cellLoc='center', colWidths=[0.3]*len(table_data[0]))
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.suptitle('Continuous CP: τ Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'tau_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved τ analysis to: {save_path}")
        plt.close()
    
    def plot_noise_effects(self, noise_results: Dict):
        """
        Plot effects of different noise models
        
        Args:
            noise_results: Results for different noise models
        """
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        noise_types = list(noise_results.keys())[:8]
        
        for idx, noise_type in enumerate(noise_types):
            ax = fig.add_subplot(gs[idx // 4, idx % 4])
            ax.set_title(f'{noise_type}', fontsize=10)
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 30)
            ax.set_aspect('equal')
            
            # Plot true obstacles
            if 'true_obs' in noise_results[noise_type]:
                self.plot_environment(ax, noise_results[noise_type]['true_obs'],
                                    color='lightgray', alpha=0.3)
            
            # Plot perceived obstacles
            if 'perceived_obs' in noise_results[noise_type]:
                self.plot_environment(ax, noise_results[noise_type]['perceived_obs'],
                                    color='blue', alpha=0.5)
            
            # Add statistics
            if 'stats' in noise_results[noise_type]:
                stats = noise_results[noise_type]['stats']
                text = f"Score: {stats.get('score', 0):.2f}"
                ax.text(2, 28, text, fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                       verticalalignment='top')
        
        plt.suptitle('Noise Model Effects', fontsize=14, fontweight='bold')
        
        save_path = os.path.join(self.save_dir, 'noise_effects.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise effects to: {save_path}")
        plt.close()
    
    def plot_ablation_results(self, ablation_data: Dict):
        """
        Plot ablation study results
        
        Args:
            ablation_data: Ablation study data
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Collision rate vs noise level with confidence intervals
        if 'noise_levels' in ablation_data:
            ax = axes[0, 0]
            ax.set_title('Collision Rate vs Noise Level', fontweight='bold')
            
            for method, data in ablation_data['noise_levels'].items():
                levels = data['levels']
                rates = data['collision_rates']
                
                # Plot with error bars if confidence intervals are available
                if 'collision_rates_ci' in data:
                    ci_data = data['collision_rates_ci']
                    # Ensure no negative error values
                    yerr_lower = [max(0, rates[i] - ci[0]) for i, ci in enumerate(ci_data)]
                    yerr_upper = [max(0, ci[1] - rates[i]) for i, ci in enumerate(ci_data)]
                    yerr = [yerr_lower, yerr_upper]
                    
                    ax.errorbar(levels, rates, yerr=yerr, marker='o', 
                              label=method, linewidth=2, markersize=6, capsize=4)
                else:
                    ax.plot(levels, rates, 'o-',
                           label=method, linewidth=2, markersize=6)
            
            # Add 5% guarantee line for CP
            ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, 
                      label='5% CP Guarantee')
            
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Collision Rate (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        
        # Plot 2: Path length vs noise level
        if 'noise_levels' in ablation_data:
            ax = axes[0, 1]
            ax.set_title('Path Length vs Noise Level', fontweight='bold')
            
            for method, data in ablation_data['noise_levels'].items():
                if 'path_lengths' in data:
                    ax.plot(data['levels'], data['path_lengths'], 'o-',
                           label=method, linewidth=2, markersize=6)
            
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Average Path Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Collision-free success rate vs noise level
        if 'noise_levels' in ablation_data:
            ax = axes[0, 2]
            ax.set_title('Collision-Free Success Rate vs Noise Level', fontweight='bold')
            
            for method, data in ablation_data['noise_levels'].items():
                # Use collision_free_success_rates instead of success_rates
                if 'collision_free_success_rates' in data:
                    ax.plot(data['levels'], data['collision_free_success_rates'], 'o-',
                           label=method, linewidth=2, markersize=6)
                elif 'path_found_rates' in data:
                    # Fallback to path_found_rates if collision_free not available
                    ax.plot(data['levels'], data['path_found_rates'], 'o--',
                           label=f"{method} (path found)", linewidth=2, markersize=6, alpha=0.5)
            
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Collision-Free Success Rate (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
        
        # Plot 4: Computation time comparison
        if 'computation_times' in ablation_data:
            ax = axes[1, 0]
            ax.set_title('Computation Time', fontweight='bold')
            
            methods = list(ablation_data['computation_times'].keys())
            times = [d['mean_time'] for d in ablation_data['computation_times'].values()]
            
            bars = ax.bar(methods, times, color=['blue', 'green', 'orange'])
            ax.set_ylabel('Time (ms)')
            ax.set_xticklabels(methods, rotation=45, ha='right')
            
            for bar, time in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{time:.1f}', ha='center', va='bottom')
        
        # Plot 5: Coverage guarantee validation
        if 'coverage_validation' in ablation_data:
            ax = axes[1, 1]
            ax.set_title('Coverage Guarantee Validation', fontweight='bold')
            
            methods = list(ablation_data['coverage_validation'].keys())
            actual = [d['actual'] for d in ablation_data['coverage_validation'].values()]
            target = [d['target'] for d in ablation_data['coverage_validation'].values()]
            
            x = np.arange(len(methods))
            width = 0.35
            
            ax.bar(x - width/2, actual, width, label='Actual', color='blue', alpha=0.7)
            ax.bar(x + width/2, target, width, label='Target', color='green', alpha=0.7)
            
            ax.set_ylabel('Coverage (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.legend()
            ax.axhline(y=95, color='red', linestyle='--', alpha=0.5)
        
        # Plot 6: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        if 'summary' in ablation_data:
            summary_text = ablation_data['summary']
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        
        plt.suptitle('Ablation Study Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, 'ablation_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation results to: {save_path}")
        plt.close()