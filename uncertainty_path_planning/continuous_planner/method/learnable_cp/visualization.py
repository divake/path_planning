#!/usr/bin/env python3
"""
Visualization Module for Learnable Conformal Prediction
Creates visualizations for analysis and ablation studies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
import os


class LearnableCPVisualizer:
    """
    Visualizer for Learnable CP results and ablation studies.
    """
    
    def __init__(self, save_dir: str = 'visualization'):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_adaptive_tau_path(self,
                              path: np.ndarray,
                              tau_values: List[float],
                              occupancy_grid: np.ndarray,
                              origin: np.ndarray,
                              resolution: float,
                              title: str = "Adaptive Tau Along Path"):
        """
        Visualize path with color-coded adaptive tau values.
        
        Args:
            path: Path waypoints [N, 2]
            tau_values: Tau value for each waypoint
            occupancy_grid: Occupancy grid
            origin: Map origin
            resolution: Map resolution
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Path with tau color coding
        ax1.imshow(occupancy_grid, cmap='gray_r', origin='lower',
                  extent=[origin[0], origin[0] + occupancy_grid.shape[1] * resolution,
                         origin[1], origin[1] + occupancy_grid.shape[0] * resolution])
        
        # Create line segments with colors based on tau
        segments = []
        for i in range(len(path) - 1):
            segments.append([path[i], path[i+1]])
        
        # Normalize tau for color mapping
        tau_array = np.array(tau_values[:-1])  # One less than path points
        norm = plt.Normalize(vmin=tau_array.min(), vmax=tau_array.max())
        
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(tau_array)
        lc.set_linewidth(2)
        
        ax1.add_collection(lc)
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax1)
        cbar.set_label('Tau (m)', rotation=270, labelpad=15)
        
        # Mark start and goal
        ax1.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='Goal')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Path with Adaptive Tau')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tau values along path
        path_distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        path_distances = np.insert(path_distances, 0, 0)
        
        ax2.plot(path_distances, tau_values, 'b-', linewidth=2)
        ax2.fill_between(path_distances, 0, tau_values, alpha=0.3)
        
        # Add horizontal lines for reference
        ax2.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Default (0.3m)')
        ax2.axhline(y=np.mean(tau_values), color='r', linestyle='--', alpha=0.5, 
                   label=f'Mean ({np.mean(tau_values):.3f}m)')
        
        ax2.set_xlabel('Distance along path (m)')
        ax2.set_ylabel('Tau (m)')
        ax2.set_title('Tau Profile Along Path')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'adaptive_tau_path.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_scores: np.ndarray,
                               title: str = "Feature Importance Analysis"):
        """
        Plot feature importance from ablation study.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance score for each feature
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(sorted_names))
        colors = plt.cm.RdYlGn(sorted_scores / sorted_scores.max())
        
        bars = ax.barh(y_pos, sorted_scores, color=colors)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'feature_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self,
                           training_history: Dict,
                           title: str = "Training Progress"):
        """
        Plot training and validation curves.
        
        Args:
            training_history: Dictionary with loss and metric histories
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        epochs = range(1, len(training_history['train_loss']) + 1)
        ax.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in training_history:
            ax.plot(epochs, training_history['val_loss'], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Coverage
        ax = axes[0, 1]
        if 'coverage' in training_history:
            ax.plot(epochs, training_history['coverage'], 'g-', linewidth=2)
            ax.axhline(y=0.9, color='r', linestyle='--', label='Target (90%)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Coverage')
            ax.set_title('Coverage Guarantee')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Average Tau
        ax = axes[1, 0]
        if 'avg_tau' in training_history:
            ax.plot(epochs, training_history['avg_tau'], 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Tau (m)')
            ax.set_title('Average Predicted Tau')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate (if available)
        ax = axes[1, 1]
        if 'lr' in training_history:
            ax.plot(epochs, training_history['lr'], 'orange', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_tau_distribution(self,
                            tau_values_dict: Dict[str, List[float]],
                            title: str = "Tau Distribution Comparison"):
        """
        Plot distribution of tau values for different methods/conditions.
        
        Args:
            tau_values_dict: Dictionary mapping method names to tau values
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Histograms
        for name, values in tau_values_dict.items():
            ax1.hist(values, bins=30, alpha=0.5, label=name, density=True)
        
        ax1.set_xlabel('Tau (m)')
        ax1.set_ylabel('Density')
        ax1.set_title('Tau Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plots
        data_for_box = [values for values in tau_values_dict.values()]
        labels_for_box = list(tau_values_dict.keys())
        
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_box)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Tau (m)')
        ax2.set_title('Tau Distribution Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        means = [np.mean(values) for values in data_for_box]
        ax2.scatter(range(1, len(means) + 1), means, color='red', 
                   marker='D', s=50, zorder=3, label='Mean')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'tau_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_ablation_results(self,
                            ablation_data: pd.DataFrame,
                            metric: str = 'coverage',
                            title: str = "Ablation Study Results"):
        """
        Plot ablation study results.
        
        Args:
            ablation_data: DataFrame with ablation results
            metric: Metric to plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by ablation type
        grouped = ablation_data.groupby('ablation_type')[metric].agg(['mean', 'std'])
        
        # Create bar plot with error bars
        x_pos = np.arange(len(grouped))
        bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'],
                     capsize=5, alpha=0.7, edgecolor='black')
        
        # Color bars based on performance
        baseline = grouped.loc['baseline', 'mean'] if 'baseline' in grouped.index else grouped['mean'].mean()
        colors = ['green' if val >= baseline else 'red' for val in grouped['mean']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, grouped['mean'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + grouped['std'].iloc[i],
                   f'{val:.3f}', ha='center', va='bottom')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.axhline(y=baseline, color='blue', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, f'ablation_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_environment_comparison(self,
                                   results_dict: Dict[str, Dict],
                                   metrics: List[str] = ['coverage', 'avg_tau', 'path_length'],
                                   title: str = "Performance Across Environments"):
        """
        Compare performance across different environments.
        
        Args:
            results_dict: Dictionary mapping environment names to results
            metrics: Metrics to compare
            title: Plot title
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        env_names = list(results_dict.keys())
        
        for ax, metric in zip(axes, metrics):
            values = [results_dict[env].get(metric, 0) for env in env_names]
            
            bars = ax.bar(range(len(env_names)), values, alpha=0.7)
            
            # Color code by performance
            if metric == 'coverage':
                colors = ['green' if v >= 0.85 else 'orange' if v >= 0.75 else 'red' 
                         for v in values]
            else:
                colors = plt.cm.viridis(np.array(values) / max(values))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(range(len(env_names)))
            ax.set_xticklabels(env_names, rotation=45, ha='right')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Environment')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'environment_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_noise_impact(self,
                         noise_results: Dict[str, Dict],
                         title: str = "Impact of Different Noise Types"):
        """
        Visualize impact of different noise types on tau prediction.
        
        Args:
            noise_results: Dictionary mapping noise type to results
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        noise_types = list(noise_results.keys())
        
        # Plot 1: Average tau by noise type
        ax = axes[0, 0]
        avg_taus = [noise_results[nt].get('avg_tau', 0) for nt in noise_types]
        bars = ax.bar(noise_types, avg_taus, color='skyblue', edgecolor='navy')
        ax.set_ylabel('Average Tau (m)')
        ax.set_title('Average Tau by Noise Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Coverage by noise type
        ax = axes[0, 1]
        coverages = [noise_results[nt].get('coverage', 0) for nt in noise_types]
        bars = ax.bar(noise_types, coverages, color='lightgreen', edgecolor='darkgreen')
        ax.axhline(y=0.9, color='r', linestyle='--', label='Target')
        ax.set_ylabel('Coverage')
        ax.set_title('Coverage by Noise Type')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Violation rate by noise type
        ax = axes[1, 0]
        violations = [noise_results[nt].get('violation_rate', 0) for nt in noise_types]
        bars = ax.bar(noise_types, violations, color='salmon', edgecolor='darkred')
        ax.set_ylabel('Violation Rate')
        ax.set_title('Safety Violations by Noise Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Efficiency (path length increase)
        ax = axes[1, 1]
        efficiencies = [noise_results[nt].get('path_length_increase', 0) for nt in noise_types]
        bars = ax.bar(noise_types, efficiencies, color='gold', edgecolor='orange')
        ax.set_ylabel('Path Length Increase (%)')
        ax.set_title('Efficiency Impact by Noise Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.save_dir, 'noise_impact.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_summary_figure(self,
                            learnable_results: Dict,
                            standard_results: Dict,
                            naive_results: Dict,
                            title: str = "Method Comparison Summary"):
        """
        Create comprehensive comparison figure.
        
        Args:
            learnable_results: Results from Learnable CP
            standard_results: Results from Standard CP
            naive_results: Results from Naive method
            title: Plot title
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        methods = ['Naive', 'Standard CP', 'Learnable CP']
        results = [naive_results, standard_results, learnable_results]
        
        # Main comparison metrics
        metrics = ['d0', 'd_avg', 'p0', 'path_length', 'computation_time']
        metric_labels = ['Initial Clearance (m)', 'Avg Clearance (m)', 
                        'Danger Zone (%)', 'Path Length (m)', 'Time (s)']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            
            values = [r.get(metric, 0) for r in results]
            bars = ax.bar(methods, values, alpha=0.7)
            
            # Color based on metric
            if metric in ['d0', 'd_avg']:
                colors = ['red' if v < 0.3 else 'green' for v in values]
            elif metric == 'p0':
                colors = ['red' if v > 10 else 'green' for v in values]
            else:
                colors = plt.cm.Blues(np.array(values) / max(values))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom')
        
        # Tau adaptation visualization
        ax = fig.add_subplot(gs[2, :])
        if 'tau_profile' in learnable_results:
            distances = learnable_results['tau_profile']['distances']
            learnable_tau = learnable_results['tau_profile']['values']
            standard_tau = [standard_results.get('tau', 0.3)] * len(distances)
            
            ax.plot(distances, learnable_tau, 'b-', linewidth=2, label='Learnable CP')
            ax.plot(distances, standard_tau, 'r--', linewidth=2, label='Standard CP')
            ax.fill_between(distances, 0, learnable_tau, alpha=0.3, color='blue')
            
            ax.set_xlabel('Distance along path (m)')
            ax.set_ylabel('Tau (m)')
            ax.set_title('Adaptive vs Fixed Tau')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        # Save
        save_path = os.path.join(self.save_dir, 'method_comparison_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def visualize_ablation_study(ablation_results: Dict, save_dir: str = 'ablation_viz'):
    """
    Create comprehensive ablation study visualizations.
    
    Args:
        ablation_results: Dictionary with ablation study results
        save_dir: Directory to save visualizations
    """
    viz = LearnableCPVisualizer(save_dir)
    
    # 1. Feature importance
    if 'feature_importance' in ablation_results:
        viz.plot_feature_importance(
            ablation_results['feature_importance']['names'],
            ablation_results['feature_importance']['scores'],
            title="Feature Importance from Ablation Study"
        )
    
    # 2. Training curves
    if 'training_history' in ablation_results:
        viz.plot_training_curves(
            ablation_results['training_history'],
            title="Model Training Progress"
        )
    
    # 3. Tau distributions
    if 'tau_distributions' in ablation_results:
        viz.plot_tau_distribution(
            ablation_results['tau_distributions'],
            title="Tau Distribution: Learnable vs Standard CP"
        )
    
    # 4. Noise impact
    if 'noise_impact' in ablation_results:
        viz.plot_noise_impact(
            ablation_results['noise_impact'],
            title="Adaptive Tau Response to Noise Types"
        )
    
    # 5. Environment comparison
    if 'environment_results' in ablation_results:
        viz.plot_environment_comparison(
            ablation_results['environment_results'],
            title="Performance Across MRPB Environments"
        )
    
    print(f"Ablation study visualizations saved to {save_dir}")