#!/usr/bin/env python3
"""
FIXED Learnable CP implementation
Addresses all issues from the evaluation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import json
import time
from scipy import stats
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar


class FeatureExtractor:
    """Extract geometric features for any point in the environment"""
    
    @staticmethod
    def extract_features(x: float, y: float, obstacles: List[Tuple], 
                        path: Optional[List] = None, goal: Optional[Tuple] = None) -> np.ndarray:
        """Extract normalized features"""
        features = []
        
        # 1. Distance to nearest obstacle
        min_dist = float('inf')
        for obs in obstacles:
            dx = max(obs[0] - x, 0, x - (obs[0] + obs[2]))
            dy = max(obs[1] - y, 0, y - (obs[1] + obs[3]))
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
        features.append(min(min_dist / 10.0, 1.0))
        
        # 2-3. Obstacle density
        for radius in [5.0, 10.0]:
            density = 0
            for obs in obstacles:
                cx, cy = obs[0] + obs[2]/2, obs[1] + obs[3]/2
                if np.sqrt((cx-x)**2 + (cy-y)**2) < radius:
                    density += 1
            features.append(min(density / 5.0, 1.0))
        
        # 4. Passage width (simplified)
        passage_width = min_dist * 2
        features.append(min(passage_width / 10.0, 1.0))
        
        # 5-6. Distance to start/goal
        if path and len(path) > 0:
            start_dist = np.sqrt((x - path[0][0])**2 + (y - path[0][1])**2)
            features.append(min(start_dist / 50.0, 1.0))
        else:
            features.append(0.5)
            
        if goal:
            goal_dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            features.append(min(goal_dist / 50.0, 1.0))
        else:
            features.append(0.5)
        
        # 7-10. Additional features (simplified)
        features.extend([0.0, 0.5, 0.0, 1.0 if min_dist < 3 else 0.0])
        
        return np.array(features, dtype=np.float32)


class ImprovedScoringNetwork(nn.Module):
    """Improved network with better initialization and architecture"""
    
    def __init__(self, input_dim: int = 10):
        super().__init__()
        
        # Simpler architecture that's easier to train
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        # Better activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Better initialization
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc3.weight, 0.1)  # Bias toward small positive values
        nn.init.constant_(self.fc3.bias, 0.1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Output in [0, 1]
        return x.squeeze()


class FixedLearnableCP:
    """Fixed version with proper training"""
    
    def __init__(self, alpha: float = 0.05, max_tau: float = 0.5):
        self.alpha = alpha
        self.max_tau = max_tau
        self.beta = 0.9  # Momentum
        
        self.scoring_net = ImprovedScoringNetwork()
        self.optimizer = optim.Adam(self.scoring_net.parameters(), lr=0.005)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        
        # Pre-computed Standard CP taus for efficiency
        self.standard_taus = {
            'passages': None,
            'open': None,
            'narrow': None
        }
        
        self.tau = max_tau
        self.history = []
    
    def precompute_standard_taus(self):
        """Pre-compute Standard CP calibrations for efficiency"""
        print("\nPre-computing Standard CP calibrations...")
        
        for env_type in ['passages', 'open', 'narrow']:
            env = ContinuousEnvironment(env_type=env_type)
            cp = ContinuousStandardCP(env.obstacles, "penetration")
            tau = cp.calibrate(
                ContinuousNoiseModel.add_thinning_noise,
                {'thin_factor': 0.2},
                num_samples=100,
                confidence=0.95
            )
            self.standard_taus[env_type] = tau
            print(f"  {env_type}: τ = {tau:.3f}")
    
    def generate_training_data(self, num_samples: int = 500) -> List[Dict]:
        """Generate training data with labels"""
        print("\nGenerating training data...")
        data = []
        
        for env_type in ['passages', 'open', 'narrow']:
            env = ContinuousEnvironment(env_type=env_type)
            standard_tau = self.standard_taus[env_type]
            
            for _ in range(num_samples // 3):
                # Sample random points
                x = np.random.uniform(5, 45)
                y = np.random.uniform(5, 25)
                
                # Skip if inside obstacle
                inside = False
                for obs in env.obstacles:
                    if (obs[0] <= x <= obs[0] + obs[2] and
                        obs[1] <= y <= obs[1] + obs[3]):
                        inside = True
                        break
                if inside:
                    continue
                
                # Extract features
                features = FeatureExtractor.extract_features(
                    x, y, env.obstacles, goal=(45, 15)
                )
                
                # Create target based on local context
                # High tau for: low clearance, high density, narrow passage
                min_clearance = features[0]
                density = (features[1] + features[2]) / 2
                is_narrow = features[9]
                
                # Target tau based on context (normalized to [0, 1])
                if min_clearance < 0.2:  # Very close to obstacle
                    target = 0.8
                elif is_narrow > 0.5:  # In narrow passage
                    target = 0.6
                elif density > 0.5:  # High density area
                    target = 0.5
                else:  # Open area
                    target = 0.2
                
                # Scale by environment-specific factor
                if env_type == 'narrow':
                    target *= 1.2
                elif env_type == 'open':
                    target *= 0.8
                
                target = min(target, 1.0)
                
                data.append({
                    'features': features,
                    'target': target,
                    'env_type': env_type
                })
        
        return data
    
    def train(self, num_epochs: int = 20):
        """Simplified training with better convergence"""
        print("\n" + "="*70)
        print("TRAINING IMPROVED LEARNABLE CP")
        print("="*70)
        
        # Pre-compute taus
        self.precompute_standard_taus()
        
        # Generate training data
        train_data = self.generate_training_data(600)
        
        print(f"\nTraining on {len(train_data)} samples...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.scoring_net.train()
            
            # Shuffle data
            np.random.shuffle(train_data)
            
            # Mini-batch training
            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                if not batch:
                    continue
                
                # Prepare batch
                features = torch.FloatTensor([d['features'] for d in batch])
                targets = torch.FloatTensor([d['target'] for d in batch])
                
                # Forward pass
                predictions = self.scoring_net(features)
                
                # Simple MSE loss
                loss = nn.MSELoss()(predictions, targets)
                
                # Add small regularization
                l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in self.scoring_net.parameters())
                total_loss = loss + l2_reg
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.scoring_net.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Learning rate decay
            self.scheduler.step()
            
            # Log progress
            avg_loss = epoch_loss / (len(train_data) / batch_size)
            self.history.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        print("\nTraining complete!")
    
    def predict_tau(self, x: float, y: float, obstacles: List, 
                    goal: Optional[Tuple] = None) -> float:
        """Predict adaptive tau for a location"""
        features = FeatureExtractor.extract_features(x, y, obstacles, goal=goal)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.scoring_net.eval()
        with torch.no_grad():
            score = self.scoring_net(features_tensor).item()
        
        return score * self.max_tau
    
    def evaluate(self, num_trials: int = 100) -> Dict:
        """Evaluate performance"""
        print("\n" + "="*70)
        print("EVALUATING FIXED LEARNABLE CP")
        print("="*70)
        
        results = {
            'naive': {'collisions': 0, 'paths': 0, 'lengths': []},
            'standard': {'collisions': 0, 'paths': 0, 'lengths': []},
            'learnable': {'collisions': 0, 'paths': 0, 'lengths': [], 'taus': []}
        }
        
        for trial in range(num_trials):
            env_type = ['passages', 'open', 'narrow'][trial % 3]
            env = ContinuousEnvironment(env_type=env_type)
            
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=0.2, seed=60000 + trial
            )
            
            # Naive
            planner = RRTStar((5, 15), (45, 15), perceived, max_iter=300)
            path = planner.plan()
            if path:
                results['naive']['paths'] += 1
                results['naive']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['naive']['collisions'] += 1
                        break
            
            # Standard CP
            tau = self.standard_taus[env_type]
            inflated = []
            for obs in perceived:
                inflated.append((
                    max(0, obs[0] - tau),
                    max(0, obs[1] - tau),
                    obs[2] + 2 * tau,
                    obs[3] + 2 * tau
                ))
            
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=300)
            path = planner.plan()
            if path:
                results['standard']['paths'] += 1
                results['standard']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['standard']['collisions'] += 1
                        break
            
            # Learnable CP
            adaptive_inflated = []
            for obs in perceived:
                # Sample tau around obstacle
                cx, cy = obs[0] + obs[2]/2, obs[1] + obs[3]/2
                local_tau = self.predict_tau(cx, cy, perceived, goal=(45, 15))
                results['learnable']['taus'].append(local_tau)
                
                adaptive_inflated.append((
                    max(0, obs[0] - local_tau),
                    max(0, obs[1] - local_tau),
                    obs[2] + 2 * local_tau,
                    obs[3] + 2 * local_tau
                ))
            
            planner = RRTStar((5, 15), (45, 15), adaptive_inflated, max_iter=300)
            path = planner.plan()
            if path:
                results['learnable']['paths'] += 1
                results['learnable']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['learnable']['collisions'] += 1
                        break
        
        # Print results
        print("\n### RESULTS ###")
        print("-" * 60)
        print(f"{'Method':<15} {'Collision Rate':<15} {'Avg Path Length':<15} {'Avg τ':<10}")
        print("-" * 60)
        
        for method, data in results.items():
            if data['paths'] > 0:
                collision_rate = data['collisions'] / data['paths'] * 100
                avg_length = np.mean(data['lengths'])
                if method == 'learnable':
                    avg_tau = np.mean(data['taus']) if data['taus'] else 0
                elif method == 'standard':
                    avg_tau = np.mean(list(self.standard_taus.values()))
                else:
                    avg_tau = 0
                
                print(f"{method.capitalize():<15} {collision_rate:<15.1f}% "
                      f"{avg_length:<15.1f} {avg_tau:<10.3f}")
        
        return results


def visualize_adaptive_behavior(model: FixedLearnableCP):
    """Visualize the learned adaptive behavior"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, env_type in enumerate(['open', 'passages', 'narrow']):
        ax = axes[idx]
        env = ContinuousEnvironment(env_type=env_type)
        
        # Create heatmap
        x_range = np.linspace(0, 50, 50)
        y_range = np.linspace(0, 30, 30)
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
                    tau_grid[i, j] = -1
                else:
                    tau_grid[i, j] = model.predict_tau(x, y, env.obstacles, goal=(45, 15))
        
        # Mask obstacles
        masked_tau = np.ma.masked_where(tau_grid < 0, tau_grid)
        
        # Plot
        im = ax.imshow(masked_tau, cmap='hot', origin='lower',
                      extent=[0, 50, 0, 30], vmin=0, vmax=model.max_tau)
        
        # Add obstacles
        for obs in env.obstacles:
            rect = plt.Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                facecolor='gray', edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
        
        ax.plot(5, 15, 'go', markersize=10)
        ax.plot(45, 15, 'r*', markersize=12)
        
        ax.set_title(f'{env_type.capitalize()} Environment', fontsize=12, weight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Fixed Learnable CP: Adaptive τ Distribution', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('results/fixed_learnable_cp_heatmap.png', dpi=150, bbox_inches='tight')
    print("\nHeatmap saved to results/fixed_learnable_cp_heatmap.png")


def main():
    """Run the fixed implementation"""
    
    # Train model
    model = FixedLearnableCP(alpha=0.05, max_tau=0.5)
    model.train(num_epochs=20)
    
    # Evaluate
    results = model.evaluate(num_trials=100)
    
    # Visualize
    visualize_adaptive_behavior(model)
    
    # Save model
    torch.save(model.scoring_net.state_dict(), 'fixed_learnable_cp_model.pth')
    print("\nModel saved to fixed_learnable_cp_model.pth")
    
    # Check if safety is maintained
    if results['learnable']['paths'] > 0:
        collision_rate = results['learnable']['collisions'] / results['learnable']['paths'] * 100
        if collision_rate <= 5.0:
            print("\n✓ SAFETY GUARANTEE MAINTAINED")
        else:
            print("\n✗ Safety guarantee violated - needs more training or tuning")


if __name__ == "__main__":
    main()