#!/usr/bin/env python3
"""
Comprehensive visualization of all three methods
Shows adaptive tau behavior and path differences
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar
from learnable_cp_proper import ProperLearnableCP, PathPlanningFeatureExtractor


def visualize_comprehensive_comparison():
    """Create comprehensive visualization of all methods"""
    
    # Load trained model
    model = ProperLearnableCP(alpha=0.05, max_tau=0.5)
    model.scoring_net.load_state_dict(torch.load('proper_learnable_cp_model.pth'))
    model.precompute_baselines()
    
    # Setup figure with 3x3 grid
    fig = plt.figure(figsize=(20, 20))
    
    # Three environments
    env_types = ['passages', 'open', 'narrow']
    
    for row, env_type in enumerate(env_types):
        env = ContinuousEnvironment(env_type=env_type)
        
        # Generate perceived obstacles (with noise)
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=80000
        )
        
        # Column 1: Naive method
        ax1 = plt.subplot(3, 3, row * 3 + 1)
        
        # Plan with naive method
        planner = RRTStar((5, 15), (45, 15), perceived, max_iter=500)
        naive_path = planner.plan()
        
        # Visualize environment
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='darkred', alpha=0.7, label='True' if obs == env.obstacles[0] else None)
            ax1.add_patch(rect)
        
        for obs in perceived:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='none', edgecolor='blue', linewidth=1.5,
                                label='Perceived' if obs == perceived[0] else None)
            ax1.add_patch(rect)
        
        if naive_path:
            path_array = np.array(naive_path)
            ax1.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, label='Path')
            
            # Check collisions
            collisions = []
            for p in naive_path:
                if env.point_in_obstacle(p[0], p[1]):
                    collisions.append(p)
            if collisions:
                collision_array = np.array(collisions)
                ax1.scatter(collision_array[:, 0], collision_array[:, 1], 
                          c='red', s=50, marker='x', label='Collision')
        
        ax1.plot(5, 15, 'go', markersize=10, label='Start')
        ax1.plot(45, 15, 'r*', markersize=15, label='Goal')
        ax1.set_xlim(0, 50)
        ax1.set_ylim(0, 30)
        ax1.set_title(f'{env_type.capitalize()}: Naive Method', fontsize=11, weight='bold')
        ax1.grid(True, alpha=0.3)
        if row == 0:
            ax1.legend(loc='upper right', fontsize=8)
        
        # Column 2: Standard CP
        ax2 = plt.subplot(3, 3, row * 3 + 2)
        
        # Get standard tau
        tau = model.baseline_taus[env_type]
        
        # Inflate obstacles uniformly
        standard_inflated = []
        for obs in perceived:
            standard_inflated.append((
                max(0, obs[0] - tau),
                max(0, obs[1] - tau),
                obs[2] + 2 * tau,
                obs[3] + 2 * tau
            ))
        
        # Plan with standard CP
        planner = RRTStar((5, 15), (45, 15), standard_inflated, max_iter=500)
        standard_path = planner.plan()
        
        # Visualize
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='darkred', alpha=0.7)
            ax2.add_patch(rect)
        
        for obs in perceived:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='none', edgecolor='blue', linewidth=1.5)
            ax2.add_patch(rect)
        
        for obs in standard_inflated:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='yellow', alpha=0.3, 
                                edgecolor='orange', linewidth=1,
                                label='Inflated' if obs == standard_inflated[0] else None)
            ax2.add_patch(rect)
        
        if standard_path:
            path_array = np.array(standard_path)
            ax2.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, label='Path')
        
        ax2.plot(5, 15, 'go', markersize=10)
        ax2.plot(45, 15, 'r*', markersize=15)
        ax2.set_xlim(0, 50)
        ax2.set_ylim(0, 30)
        ax2.set_title(f'{env_type.capitalize()}: Standard CP (τ={tau:.3f})', 
                     fontsize=11, weight='bold')
        ax2.grid(True, alpha=0.3)
        if row == 0:
            ax2.legend(loc='upper right', fontsize=8)
        
        # Column 3: Learnable CP
        ax3 = plt.subplot(3, 3, row * 3 + 3)
        
        # Get adaptive tau for each obstacle
        obstacle_centers = [(obs[0] + obs[2]/2, obs[1] + obs[3]/2) for obs in perceived]
        tau_values = model.predict_tau_vectorized(obstacle_centers, perceived)
        
        # Adaptive inflation
        learnable_inflated = []
        for obs, local_tau in zip(perceived, tau_values):
            learnable_inflated.append((
                max(0, obs[0] - local_tau),
                max(0, obs[1] - local_tau),
                obs[2] + 2 * local_tau,
                obs[3] + 2 * local_tau
            ))
        
        # Plan with learnable CP
        planner = RRTStar((5, 15), (45, 15), learnable_inflated, max_iter=500)
        learnable_path = planner.plan()
        
        # Visualize
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='darkred', alpha=0.7)
            ax3.add_patch(rect)
        
        for obs in perceived:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='none', edgecolor='blue', linewidth=1.5)
            ax3.add_patch(rect)
        
        # Color-code inflation by tau value
        for obs, local_tau in zip(learnable_inflated, tau_values):
            # Color based on tau magnitude
            color_intensity = local_tau / 0.5  # Normalize by max_tau
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor=plt.cm.YlOrRd(color_intensity), 
                                alpha=0.3,
                                edgecolor='orange', linewidth=1)
            ax3.add_patch(rect)
        
        if learnable_path:
            path_array = np.array(learnable_path)
            ax3.plot(path_array[:, 0], path_array[:, 1], 'g-', linewidth=2, label='Path')
        
        ax3.plot(5, 15, 'go', markersize=10)
        ax3.plot(45, 15, 'r*', markersize=15)
        ax3.set_xlim(0, 50)
        ax3.set_ylim(0, 30)
        
        # Add text showing tau range
        tau_min, tau_max = np.min(tau_values), np.max(tau_values)
        ax3.set_title(f'{env_type.capitalize()}: Learnable CP (τ∈[{tau_min:.2f}, {tau_max:.2f}])', 
                     fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for tau values
        if row == 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                       norm=plt.Normalize(vmin=0, vmax=0.5))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax3, fraction=0.046, pad=0.04)
            cbar.set_label('τ value', fontsize=9)
    
    plt.suptitle('Comprehensive Comparison: Naive vs Standard CP vs Learnable CP', 
                fontsize=16, weight='bold', y=0.98)
    
    # Add method summary at bottom
    fig.text(0.5, 0.02, 
            'Naive: No safety margin | Standard CP: Uniform τ | Learnable CP: Adaptive τ based on local context',
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('results/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to results/comprehensive_comparison.png")
    
    # Create second figure: Tau heatmap
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, env_type in enumerate(env_types):
        ax = axes[idx]
        env = ContinuousEnvironment(env_type=env_type)
        
        # Create heatmap of predicted tau values
        x_range = np.linspace(0, 50, 100)
        y_range = np.linspace(0, 30, 60)
        tau_grid = np.zeros((len(y_range), len(x_range)))
        
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                # Check if inside obstacle
                inside = False
                for obs in env.obstacles:
                    if (obs[0] <= x <= obs[0] + obs[2] and
                        obs[1] <= y <= obs[1] + obs[3]):
                        inside = True
                        break
                
                if inside:
                    tau_grid[i, j] = np.nan
                else:
                    features = PathPlanningFeatureExtractor.extract_location_features(
                        x, y, env.obstacles, goal=(45, 15)
                    )
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    
                    model.scoring_net.eval()
                    with torch.no_grad():
                        score = torch.sigmoid(model.scoring_net.scoring_network(features_tensor)).item()
                    tau_grid[i, j] = score * model.max_tau
        
        # Plot heatmap
        masked_tau = np.ma.masked_invalid(tau_grid)
        im = ax.imshow(masked_tau, cmap='hot', origin='lower',
                      extent=[0, 50, 0, 30], vmin=0, vmax=model.max_tau)
        
        # Add obstacles
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='gray', edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        ax.plot(5, 15, 'go', markersize=10)
        ax.plot(45, 15, 'r*', markersize=12)
        
        ax.set_title(f'{env_type.capitalize()} Environment: Adaptive τ', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Learnable CP: Adaptive Safety Margins (τ) Heatmap', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('results/learnable_tau_heatmap.png', dpi=150, bbox_inches='tight')
    print("Tau heatmap saved to results/learnable_tau_heatmap.png")


if __name__ == "__main__":
    visualize_comprehensive_comparison()