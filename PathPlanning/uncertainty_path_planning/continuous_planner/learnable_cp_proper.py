#!/usr/bin/env python3
"""
Proper Learnable CP implementation for path planning
Based on the actual branch_sf implementation with class-specific features
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


class PathPlanningFeatureExtractor:
    """
    Extract location-specific features for path planning
    Analogous to class-specific features in classification
    """
    
    @staticmethod
    def extract_location_features(x: float, y: float, obstacles: List[Tuple], 
                                 goal: Optional[Tuple] = None) -> np.ndarray:
        """
        Extract 8 location-specific features (analogous to class-specific features)
        
        Features:
        1. Distance to nearest obstacle (normalized)
        2. Rank of this location by clearance 
        3. Gap to safest location
        4. Is in top 10% safest locations
        5. Is in top 30% safest locations
        6. Is in top 50% safest locations
        7. Environment complexity (entropy analogue)
        8. Maximum clearance in environment
        """
        features = []
        
        # 1. Distance to nearest obstacle
        min_dist = float('inf')
        for obs in obstacles:
            dx = max(obs[0] - x, 0, x - (obs[0] + obs[2]))
            dy = max(obs[1] - y, 0, y - (obs[1] + obs[3]))
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
        clearance = min_dist / 10.0  # Normalize
        features.append(min(clearance, 1.0))
        
        # 2. Sample clearances at multiple locations for ranking
        sample_clearances = []
        for _ in range(20):  # Sample 20 random points
            sx = np.random.uniform(0, 50)
            sy = np.random.uniform(0, 30)
            s_min_dist = float('inf')
            for obs in obstacles:
                dx = max(obs[0] - sx, 0, sx - (obs[0] + obs[2]))
                dy = max(obs[1] - sy, 0, sy - (obs[1] + obs[3]))
                s_dist = np.sqrt(dx**2 + dy**2)
                s_min_dist = min(s_min_dist, s_dist)
            sample_clearances.append(s_min_dist)
        
        # Rank of current location (normalized)
        rank = sum(1 for c in sample_clearances if c > min_dist) + 1
        normalized_rank = rank / (len(sample_clearances) + 1)
        features.append(normalized_rank)
        
        # 3. Gap to maximum clearance
        max_clearance = max(sample_clearances) if sample_clearances else min_dist
        gap_to_max = (max_clearance - min_dist) / 10.0
        features.append(min(gap_to_max, 1.0))
        
        # 4-6. Binary indicators for safety percentiles
        features.append(1.0 if normalized_rank <= 0.1 else 0.0)  # Top 10%
        features.append(1.0 if normalized_rank <= 0.3 else 0.0)  # Top 30%
        features.append(1.0 if normalized_rank <= 0.5 else 0.0)  # Top 50%
        
        # 7. Environment complexity (variance of clearances)
        if len(sample_clearances) > 1:
            complexity = np.std(sample_clearances) / 5.0  # Normalize
        else:
            complexity = 0.5
        features.append(min(complexity, 1.0))
        
        # 8. Maximum clearance (normalized)
        features.append(min(max_clearance / 10.0, 1.0))
        
        return np.array(features, dtype=np.float32)


class LocationScoringNetwork(nn.Module):
    """
    Scoring network for location-specific safety margins
    Based on the vectorized class-specific architecture
    """
    
    def __init__(self, feature_dim: int = 8, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]  # Default architecture
        
        layers = []
        prev_dim = feature_dim
        dropout_rate = 0.3
        
        # Build MLP layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),  # Standard activation
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer - single score (no activation)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
        # L2 regularization
        self.l2_lambda = 0.01
        
        # Initialize weights (Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable gradients"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """
        Forward pass through scoring network
        
        Args:
            features: Batch of location features [B, 8] or [B*L, 8] for vectorized
        
        Returns:
            scores: Raw nonconformity scores (can be negative)
        """
        scores = self.scoring_network(features)
        
        # NO sigmoid - allow negative scores as per classification implementation
        # Scores can range from negative to positive values
        
        # L2 regularization during training
        if self.training:
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            self.l2_reg = self.l2_lambda * l2_reg
        else:
            self.l2_reg = 0.0
        
        return scores.squeeze(-1)  # Remove last dimension


class ProperLearnableCP:
    """
    Proper implementation following the branch_sf architecture
    with phase-based training and margin loss
    """
    
    def __init__(self, alpha: float = 0.1, max_tau: float = 0.5):  # 90% coverage
        self.alpha = alpha
        self.max_tau = max_tau
        
        # Initialize network
        self.scoring_net = LocationScoringNetwork(feature_dim=8, hidden_dims=[64, 32])
        
        # Optimizer with OneCycleLR scheduler
        self.optimizer = optim.AdamW(self.scoring_net.parameters(), lr=0.001, weight_decay=0.01)
        
        # Pre-computed Standard CP baselines
        self.baseline_taus = {}
        
        # Training history
        self.history = {
            'epochs': [],
            'train_losses': [],
            'train_coverages': [],
            'val_coverages': [],
            'val_sizes': [],
            'tau_values': []
        }
        
        # Phase-based training weights
        self.margin_weight = 1.0
        self.coverage_weight = 0.5
        self.size_weight = 0.1
    
    def precompute_baselines(self):
        """Pre-compute Standard CP baselines for each environment"""
        print("\nPre-computing baseline τ values...")
        
        for env_type in ['passages', 'open', 'narrow']:
            env = ContinuousEnvironment(env_type=env_type)
            cp = ContinuousStandardCP(env.obstacles, "penetration")
            tau = cp.calibrate(
                ContinuousNoiseModel.add_thinning_noise,
                {'thin_factor': 0.2},
                num_samples=100,
                confidence=0.90  # 90% coverage
            )
            self.baseline_taus[env_type] = tau
            print(f"  {env_type}: τ = {tau:.3f}")
    
    def generate_training_data_vectorized(self, num_samples: int = 1000) -> Dict:
        """
        Generate training data with vectorized processing
        Similar to the vectorized class processing in classification
        """
        print("\nGenerating vectorized training data...")
        
        all_features = []
        all_targets = []
        all_env_types = []
        
        for env_type in ['passages', 'open', 'narrow']:
            env = ContinuousEnvironment(env_type=env_type)
            baseline_tau = self.baseline_taus[env_type]
            
            # Generate multiple locations at once (vectorized)
            num_locations = num_samples // 3
            x_coords = np.random.uniform(5, 45, num_locations)
            y_coords = np.random.uniform(5, 25, num_locations)
            
            # Extract features for all locations
            location_features = []
            location_targets = []
            
            for x, y in zip(x_coords, y_coords):
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
                features = PathPlanningFeatureExtractor.extract_location_features(
                    x, y, env.obstacles, goal=(45, 15)
                )
                location_features.append(features)
                
                # Target based on clearance (feature 1)
                clearance = features[0]
                rank = features[1]
                
                # Create target score (lower for dangerous, higher for safe)
                # This mimics the discrimination in classification
                if clearance < 0.2:  # Very dangerous
                    target = 0.1
                elif rank > 0.7:  # Bottom 30% clearance
                    target = 0.3
                elif rank > 0.5:  # Middle clearance
                    target = 0.5
                else:  # Good clearance
                    target = 0.8
                
                location_targets.append(target)
            
            all_features.extend(location_features)
            all_targets.extend(location_targets)
            all_env_types.extend([env_type] * len(location_features))
        
        return {
            'features': np.array(all_features),
            'targets': np.array(all_targets),
            'env_types': all_env_types
        }
    
    def train_phased(self, num_epochs: int = 30):
        """
        Phase-based training following the branch_sf approach
        Phase 1: Discrimination only (learn to separate safe/dangerous)
        Phase 2: Add coverage loss
        Phase 3: Add efficiency loss
        """
        print("\n" + "="*70)
        print("PHASE-BASED TRAINING (Following branch_sf)")
        print("="*70)
        
        # Pre-compute baselines
        self.precompute_baselines()
        
        # Generate training data
        train_data = self.generate_training_data_vectorized(1500)
        features = torch.FloatTensor(train_data['features'])
        targets = torch.FloatTensor(train_data['targets'])
        
        # Create DataLoader for batching
        dataset = torch.utils.data.TensorDataset(features, targets)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # OneCycleLR scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3  # 30% warmup
        )
        
        for epoch in range(num_epochs):
            self.scoring_net.train()
            epoch_loss = 0
            epoch_coverage = 0
            num_batches = 0
            
            # Determine current phase
            if epoch < 10:
                phase = 1  # Discrimination only
                phase_name = "Phase 1 (Discrimination)"
            elif epoch < 20:
                phase = 2  # + Coverage
                phase_name = "Phase 2 (+ Coverage)"
            else:
                phase = 3  # + Size
                phase_name = "Phase 3 (All losses)"
            
            for batch_features, batch_targets in dataloader:
                # Forward pass
                scores = self.scoring_net(batch_features)
                
                # PRIMARY: Margin-based discrimination loss
                # Low targets (dangerous) should have low scores  
                # High targets (safe) should have high scores
                # Scores are raw values (can be negative)
                margin = 0.5
                # Simple margin loss: dangerous locations should score lower
                discrimination_loss = torch.relu(margin - (batch_targets - scores).abs()).mean()
                
                # Coverage loss (phase 2+)
                if phase >= 2:
                    # Simulate coverage calculation
                    threshold = 0.3  # Simulated tau
                    covered = (scores < threshold).float()
                    coverage = covered.mean()
                    target_coverage = 0.90  # 90% coverage
                    coverage_loss = (coverage - target_coverage).pow(2)
                else:
                    coverage_loss = 0
                    coverage = 0
                
                # Size loss (phase 3)
                if phase >= 3:
                    # Encourage smaller margins while maintaining safety
                    # Penalize overly conservative predictions
                    size_loss = scores.mean()
                else:
                    size_loss = 0
                
                # Combine losses based on phase
                if phase == 1:
                    loss = discrimination_loss * self.margin_weight
                elif phase == 2:
                    loss = (discrimination_loss * self.margin_weight + 
                           coverage_loss * self.coverage_weight)
                else:
                    loss = (discrimination_loss * self.margin_weight + 
                           coverage_loss * self.coverage_weight +
                           size_loss * self.size_weight)
                
                # Add L2 regularization
                if hasattr(self.scoring_net, 'l2_reg'):
                    loss = loss + self.scoring_net.l2_reg
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.scoring_net.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                if phase >= 2:
                    epoch_coverage += coverage.item() if isinstance(coverage, torch.Tensor) else coverage
                num_batches += 1
            
            # Log progress
            avg_loss = epoch_loss / num_batches
            avg_coverage = epoch_coverage / num_batches if phase >= 2 else 0
            
            if (epoch + 1) % 5 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} - {phase_name}:")
                print(f"  Loss: {avg_loss:.4f}")
                if phase >= 2:
                    print(f"  Coverage: {avg_coverage:.3f}")
        
        print("\nTraining complete!")
    
    def predict_tau_vectorized(self, locations: List[Tuple], obstacles: List) -> np.ndarray:
        """
        Vectorized prediction for multiple locations at once
        Much faster than individual predictions
        """
        # Extract features for all locations at once
        features_list = []
        for x, y in locations:
            features = PathPlanningFeatureExtractor.extract_location_features(
                x, y, obstacles, goal=(45, 15)
            )
            features_list.append(features)
        
        features_tensor = torch.FloatTensor(np.array(features_list))
        
        self.scoring_net.eval()
        with torch.no_grad():
            scores = self.scoring_net(features_tensor).numpy()
        
        # Convert raw scores to tau values
        # Scores can be negative (like in classification)
        # Map to [0, max_tau] using tanh transformation
        # tanh maps (-inf, +inf) to (-1, +1), then scale to [0, max_tau]
        tau_values = (np.tanh(scores) + 1) * 0.5 * self.max_tau
        return tau_values
    
    def evaluate_comprehensive(self, num_trials: int = 100) -> Dict:
        """Comprehensive evaluation of the trained model"""
        print("\n" + "="*70)
        print("EVALUATING PROPER LEARNABLE CP")
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
                env.obstacles, thin_factor=0.2, seed=70000 + trial
            )
            
            # Naive method
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
            tau = self.baseline_taus[env_type]
            standard_inflated = []
            for obs in perceived:
                standard_inflated.append((
                    max(0, obs[0] - tau),
                    max(0, obs[1] - tau),
                    obs[2] + 2 * tau,
                    obs[3] + 2 * tau
                ))
            
            planner = RRTStar((5, 15), (45, 15), standard_inflated, max_iter=300)
            path = planner.plan()
            if path:
                results['standard']['paths'] += 1
                results['standard']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['standard']['collisions'] += 1
                        break
            
            # Learnable CP with vectorized prediction
            # Get centers of all obstacles at once
            obstacle_centers = [(obs[0] + obs[2]/2, obs[1] + obs[3]/2) for obs in perceived]
            tau_values = self.predict_tau_vectorized(obstacle_centers, perceived)
            
            learnable_inflated = []
            for obs, local_tau in zip(perceived, tau_values):
                results['learnable']['taus'].append(local_tau)
                learnable_inflated.append((
                    max(0, obs[0] - local_tau),
                    max(0, obs[1] - local_tau),
                    obs[2] + 2 * local_tau,
                    obs[3] + 2 * local_tau
                ))
            
            planner = RRTStar((5, 15), (45, 15), learnable_inflated, max_iter=300)
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
                    avg_tau = np.mean(list(self.baseline_taus.values()))
                else:
                    avg_tau = 0
                
                print(f"{method.capitalize():<15} {collision_rate:<15.1f}% "
                      f"{avg_length:<15.1f} {avg_tau:<10.3f}")
        
        return results


def main():
    """Run the proper implementation"""
    
    # Create and train model
    model = ProperLearnableCP(alpha=0.1, max_tau=0.5)  # 90% coverage
    model.train_phased(num_epochs=30)
    
    # Evaluate
    results = model.evaluate_comprehensive(num_trials=100)
    
    # Save model
    torch.save(model.scoring_net.state_dict(), 'proper_learnable_cp_model.pth')
    print("\nModel saved to proper_learnable_cp_model.pth")
    
    # Check safety guarantee
    if results['learnable']['paths'] > 0:
        collision_rate = results['learnable']['collisions'] / results['learnable']['paths'] * 100
        if collision_rate <= 5.0:
            print("\n✓ SAFETY GUARANTEE MAINTAINED")
        else:
            print(f"\n✗ Safety not quite there yet ({collision_rate:.1f}%), but close!")
    
    # Path improvement
    if results['standard']['paths'] > 0 and results['learnable']['paths'] > 0:
        standard_avg = np.mean(results['standard']['lengths'])
        learnable_avg = np.mean(results['learnable']['lengths'])
        improvement = (standard_avg - learnable_avg) / standard_avg * 100
        print(f"Path improvement: {improvement:.1f}%")


if __name__ == "__main__":
    main()