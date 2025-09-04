#!/usr/bin/env python3
"""
FINAL Learnable CP implementation addressing all concerns:
1. Seeds everywhere for reproducibility
2. Clear noise model explanation
3. 90% coverage (not 95%)
4. No sigmoid constraint (negative scores allowed)
5. More samples for calibration (500)
6. More trials for evaluation (500)
7. Principled target assignment
8. Timing measurements
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import time
from scipy import stats
import sys
import random
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar


class EnhancedNoiseModel:
    """Enhanced noise model with multiple error modes"""
    
    @staticmethod
    def add_realistic_noise(obstacles: List[Tuple], 
                           noise_type: str = 'mixed',
                           noise_level: float = 0.2,
                           seed: Optional[int] = None) -> List[Tuple]:
        """
        Add realistic perception noise
        
        noise_type:
        - 'thinning': Only shrink obstacles (current)
        - 'thickening': Only expand obstacles  
        - 'mixed': Random mix of shrink/expand/shift
        - 'gaussian': Gaussian noise on all edges
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        noisy_obstacles = []
        
        for obs in obstacles:
            x, y, width, height = obs
            
            if noise_type == 'thinning':
                # Original: only shrink
                thin_w = noise_level * width
                thin_h = noise_level * height
                noisy_obstacles.append((
                    x + thin_w/2, y + thin_h/2,
                    max(0.1, width - thin_w),
                    max(0.1, height - thin_h)
                ))
                
            elif noise_type == 'thickening':
                # Only expand
                thick_w = noise_level * width
                thick_h = noise_level * height
                noisy_obstacles.append((
                    max(0, x - thick_w/2), 
                    max(0, y - thick_h/2),
                    width + thick_w,
                    height + thick_h
                ))
                
            elif noise_type == 'mixed':
                # Randomly shrink, expand, or shift
                mode = np.random.choice(['shrink', 'expand', 'shift'])
                factor = np.random.uniform(0, noise_level)
                
                if mode == 'shrink':
                    delta_w = factor * width
                    delta_h = factor * height
                    noisy_obstacles.append((
                        x + delta_w/2, y + delta_h/2,
                        max(0.1, width - delta_w),
                        max(0.1, height - delta_h)
                    ))
                elif mode == 'expand':
                    delta_w = factor * width
                    delta_h = factor * height
                    noisy_obstacles.append((
                        max(0, x - delta_w/2),
                        max(0, y - delta_h/2),
                        width + delta_w,
                        height + delta_h
                    ))
                else:  # shift
                    shift_x = np.random.uniform(-factor*2, factor*2)
                    shift_y = np.random.uniform(-factor*2, factor*2)
                    noisy_obstacles.append((
                        x + shift_x, y + shift_y,
                        width, height
                    ))
                    
            elif noise_type == 'gaussian':
                # Gaussian noise on all parameters
                noisy_obstacles.append((
                    x + np.random.normal(0, noise_level),
                    y + np.random.normal(0, noise_level),
                    max(0.1, width + np.random.normal(0, noise_level*width)),
                    max(0.1, height + np.random.normal(0, noise_level*height))
                ))
        
        return noisy_obstacles


class PrincipledFeatureExtractor:
    """
    Feature extraction with principled design
    Based on geometric properties relevant to collision risk
    """
    
    @staticmethod
    def extract_features(x: float, y: float, obstacles: List[Tuple], 
                         goal: Optional[Tuple] = None) -> np.ndarray:
        """
        Extract 10 principled features for collision risk assessment
        """
        features = []
        
        # 1. Minimum clearance (most important)
        min_dist = float('inf')
        for obs in obstacles:
            dx = max(obs[0] - x, 0, x - (obs[0] + obs[2]))
            dy = max(obs[1] - y, 0, y - (obs[1] + obs[3]))
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
        clearance = min_dist
        features.append(clearance)
        
        # 2. Average clearance to nearest 3 obstacles
        distances = []
        for obs in obstacles:
            cx, cy = obs[0] + obs[2]/2, obs[1] + obs[3]/2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            distances.append(dist)
        distances.sort()
        avg_near_clear = np.mean(distances[:3]) if len(distances) >= 3 else clearance
        features.append(avg_near_clear)
        
        # 3. Obstacle density in 5-unit radius
        density_5 = sum(1 for obs in obstacles 
                       if np.sqrt((x - (obs[0] + obs[2]/2))**2 + 
                                 (y - (obs[1] + obs[3]/2))**2) < 5)
        features.append(density_5)
        
        # 4. Obstacle density in 10-unit radius  
        density_10 = sum(1 for obs in obstacles 
                        if np.sqrt((x - (obs[0] + obs[2]/2))**2 + 
                                  (y - (obs[1] + obs[3]/2))**2) < 10)
        features.append(density_10)
        
        # 5. Passage width estimate (clearance in 4 directions)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        passage_widths = []
        for dx, dy in directions:
            dist = 0
            for step in range(1, 20):
                test_x = x + dx * step * 0.5
                test_y = y + dy * step * 0.5
                hit = False
                for obs in obstacles:
                    if (obs[0] <= test_x <= obs[0] + obs[2] and
                        obs[1] <= test_y <= obs[1] + obs[3]):
                        hit = True
                        break
                if hit:
                    dist = step * 0.5
                    break
            passage_widths.append(dist if dist > 0 else 10)
        min_passage = min(passage_widths)
        features.append(min_passage)
        
        # 6-7. Position in environment (normalized)
        features.append(x / 50.0)  # x position
        features.append(y / 30.0)  # y position
        
        # 8. Distance to goal (if provided)
        if goal:
            goal_dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            features.append(goal_dist / 50.0)
        else:
            features.append(0.5)
        
        # 9. Is near boundary
        near_boundary = 1.0 if (x < 5 or x > 45 or y < 5 or y > 25) else 0.0
        features.append(near_boundary)
        
        # 10. Variance of nearby clearances (environment complexity)
        sample_clearances = []
        for _ in range(10):
            sx = x + np.random.uniform(-5, 5)
            sy = y + np.random.uniform(-5, 5)
            s_min_dist = float('inf')
            for obs in obstacles:
                dx = max(obs[0] - sx, 0, sx - (obs[0] + obs[2]))
                dy = max(obs[1] - sy, 0, sy - (obs[1] + obs[3]))
                s_dist = np.sqrt(dx**2 + dy**2)
                s_min_dist = min(s_min_dist, s_dist)
            sample_clearances.append(s_min_dist)
        complexity = np.std(sample_clearances) if len(sample_clearances) > 1 else 0
        features.append(complexity)
        
        return np.array(features, dtype=np.float32)


class UnconstrainedScoringNetwork(nn.Module):
    """
    Scoring network WITHOUT sigmoid constraint
    Allows negative scores as in classification
    """
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]  # Wider network
        
        # Build network with batch normalization
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Add batch norm
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:  # No dropout on last hidden layer
                layers.append(nn.Dropout(0.1))  # Less dropout
            prev_dim = hidden_dim
        
        # Output layer - NO activation (raw scores)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Careful initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.1)  # Small positive bias
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Returns raw scores (can be negative)
        Range typically [-2, +2] based on experience
        """
        return self.network(x).squeeze(-1)


class FinalLearnableCP:
    """
    Final implementation with all fixes
    """
    
    def __init__(self, coverage: float = 0.90, max_tau: float = 1.0):
        """
        Args:
            coverage: Target coverage level (0.90 = 90%)
            max_tau: Maximum allowed tau value
        """
        self.coverage = coverage
        self.alpha = 1 - coverage  # 0.1 for 90% coverage
        self.max_tau = max_tau
        
        # Set all random seeds
        self.set_seeds(42)
        
        # Initialize network
        self.scoring_net = UnconstrainedScoringNetwork(input_dim=10, hidden_dims=[128, 64, 32])
        
        # Optimizer
        self.optimizer = optim.AdamW(self.scoring_net.parameters(), 
                                     lr=0.01, weight_decay=0.001)  # Higher LR
        
        # Storage for baselines and metrics
        self.baseline_taus = {}
        self.calibration_times = {}
        self.training_time = 0
        self.inference_times = []
    
    def set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def calibrate_standard_cp(self, num_samples: int = 500):
        """
        Calibrate Standard CP with MORE samples
        Using 90% coverage (not 95%)
        """
        print(f"\nCalibrating Standard CP with {num_samples} samples (90% coverage)...")
        
        for env_type in ['passages', 'open', 'narrow']:
            start_time = time.time()
            
            env = ContinuousEnvironment(env_type=env_type)
            cp = ContinuousStandardCP(env.obstacles, "penetration")
            
            # Use 90% coverage!
            tau = cp.calibrate(
                EnhancedNoiseModel.add_realistic_noise,  # Enhanced noise
                {'noise_type': 'thinning', 'noise_level': 0.2},
                num_samples=num_samples,
                confidence=0.90  # 90% coverage
            )
            
            self.baseline_taus[env_type] = tau
            self.calibration_times[env_type] = time.time() - start_time
            
            print(f"  {env_type}: τ = {tau:.3f} (time: {self.calibration_times[env_type]:.2f}s)")
    
    def generate_principled_training_data(self, num_samples: int = 2000):
        """
        Generate training data with PRINCIPLED target assignment
        Based on actual collision risk
        """
        print(f"\nGenerating {num_samples} training samples with principled targets...")
        
        all_features = []
        all_targets = []
        all_clearances = []
        
        for env_type in ['passages', 'open', 'narrow']:
            env = ContinuousEnvironment(env_type=env_type)
            samples_per_env = num_samples // 3
            
            # Set seed for reproducible sampling
            np.random.seed(1000 + hash(env_type) % 1000)
            
            for i in range(samples_per_env):
                x = np.random.uniform(2, 48)
                y = np.random.uniform(2, 28)
                
                # Skip if inside obstacle
                inside = any(obs[0] <= x <= obs[0] + obs[2] and
                           obs[1] <= y <= obs[1] + obs[3]
                           for obs in env.obstacles)
                if inside:
                    continue
                
                # Extract features
                features = PrincipledFeatureExtractor.extract_features(
                    x, y, env.obstacles, goal=(45, 15)
                )
                
                # PRINCIPLED target based on clearance
                # Use actual clearance value for continuous target
                clearance = features[0]  # First feature is clearance
                
                # DIRECT tau target (skip score, train for tau directly)
                # Map clearance directly to desired tau
                baseline_tau = self.baseline_taus.get(env_type, 0.5)
                
                if clearance < 0.5:
                    # Very dangerous - use baseline tau or higher
                    target_tau = baseline_tau * 1.2
                elif clearance < 1.0:
                    # Dangerous - use baseline tau
                    target_tau = baseline_tau
                elif clearance < 2.0:
                    # Moderate - reduce tau
                    target_tau = baseline_tau * 0.7
                elif clearance < 4.0:
                    # Safe - minimal tau
                    target_tau = baseline_tau * 0.3
                else:
                    # Very safe - minimum tau
                    target_tau = baseline_tau * 0.1
                
                # Convert tau to score using inverse of our mapping
                # tau = max_tau / (1 + exp(score))
                # score = log(max_tau/tau - 1)
                target_tau = np.clip(target_tau, 0.01, self.max_tau - 0.01)
                target = np.log(self.max_tau / target_tau - 1)
                
                all_features.append(features)
                all_targets.append(target)
                all_clearances.append(clearance)
        
        # Print statistics
        targets_array = np.array(all_targets)
        print(f"  Target statistics: min={targets_array.min():.2f}, "
              f"max={targets_array.max():.2f}, mean={targets_array.mean():.2f}")
        print(f"  High targets (dangerous->high tau): {(targets_array > 0).sum()} "
              f"({(targets_array > 0).mean()*100:.1f}%)")
        print(f"  Low targets (safe->low tau): {(targets_array < 0).sum()} "
              f"({(targets_array < 0).mean()*100:.1f}%)")
        
        return {
            'features': np.array(all_features),
            'targets': np.array(all_targets),
            'clearances': np.array(all_clearances)
        }
    
    def train(self, num_epochs: int = 150):  # More epochs
        """
        Train with proper loss function
        No sigmoid, allowing negative scores
        """
        print(f"\n{'='*70}")
        print("TRAINING with unconstrained scores (like classification)")
        print(f"{'='*70}")
        
        start_time = time.time()
        self.set_seeds(42)  # Reset seeds for training
        
        # Generate training data
        train_data = self.generate_principled_training_data(5000)  # More data
        features = torch.FloatTensor(train_data['features'])
        targets = torch.FloatTensor(train_data['targets'])
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(features, targets)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True,
            generator=torch.Generator().manual_seed(42)  # Seeded shuffle
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            self.scoring_net.train()
            epoch_loss = 0
            
            for batch_features, batch_targets in dataloader:
                # Forward pass - raw scores
                scores = self.scoring_net(batch_features)
                
                # Loss: MSE + ranking loss
                mse_loss = nn.MSELoss()(scores, batch_targets)
                
                # Ranking loss: dangerous should score HIGHER than safe (inverted for path planning)
                n = len(scores)
                if n > 1:
                    # Sample pairs
                    idx1 = torch.randperm(n)[:n//2]
                    idx2 = torch.randperm(n)[:n//2]
                    
                    # Ranking constraint (INVERTED for path planning)
                    # Higher target (dangerous) should have higher score
                    margin = 0.5
                    ranking_loss = torch.relu(
                        scores[idx2] - scores[idx1] + 
                        margin * torch.sign(batch_targets[idx1] - batch_targets[idx2])
                    ).mean()
                else:
                    ranking_loss = 0
                
                # Weight dangerous areas more heavily
                weights = torch.ones_like(batch_targets)
                weights[batch_targets > 0] = 5.0  # 5x weight for dangerous
                weighted_mse = (weights * (scores - batch_targets).pow(2)).mean()
                
                total_loss = weighted_mse + 0.5 * ranking_loss  # More ranking
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.scoring_net.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += total_loss.item()
            
            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        self.training_time = time.time() - start_time
        print(f"\nTraining complete in {self.training_time:.2f}s")
        
        return history
    
    def predict_tau(self, x: float, y: float, obstacles: List, goal=None) -> float:
        """
        Predict tau from raw scores
        Map unbounded scores to [0, max_tau]
        
        CRITICAL: In path planning, DANGEROUS areas need HIGH tau (opposite of classification)
        - Low clearance → HIGH tau for safety
        - High clearance → LOW tau for efficiency
        """
        start_time = time.time()
        
        features = PrincipledFeatureExtractor.extract_features(x, y, obstacles, goal)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.scoring_net.eval()
        with torch.no_grad():
            raw_score = self.scoring_net(features_tensor).item()
        
        # INVERT the mapping: negative scores → high tau, positive scores → low tau
        # This is opposite of classification where true class gets low score
        # Here dangerous locations need HIGH safety margin
        tau = self.max_tau / (1 + np.exp(raw_score))  # Note: positive raw_score, not negative
        
        self.inference_times.append(time.time() - start_time)
        return tau
    
    def evaluate_comprehensive(self, num_trials: int = 500):
        """
        Comprehensive evaluation with MORE trials
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION with {num_trials} trials")
        print(f"{'='*70}")
        
        results = {
            'naive': {'collisions': 0, 'paths': 0, 'lengths': [], 'times': []},
            'standard': {'collisions': 0, 'paths': 0, 'lengths': [], 'times': []},
            'learnable': {'collisions': 0, 'paths': 0, 'lengths': [], 'times': [], 'taus': []}
        }
        
        # Test different noise levels
        noise_levels = [0.15, 0.20, 0.25]
        
        for trial in range(num_trials):
            env_type = ['passages', 'open', 'narrow'][trial % 3]
            noise_level = noise_levels[trial % len(noise_levels)]
            
            env = ContinuousEnvironment(env_type=env_type)
            
            # Use seed for reproducible noise
            seed = 100000 + trial
            perceived = EnhancedNoiseModel.add_realistic_noise(
                env.obstacles, 
                noise_type='thinning',
                noise_level=noise_level,
                seed=seed
            )
            
            # NAIVE
            start = time.time()
            # Set numpy seed for RRT* random sampling
            np.random.seed(seed)
            planner = RRTStar((5, 15), (45, 15), perceived, max_iter=500)
            path = planner.plan()
            results['naive']['times'].append(time.time() - start)
            
            if path:
                results['naive']['paths'] += 1
                results['naive']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['naive']['collisions'] += 1
                        break
            
            # STANDARD CP
            start = time.time()
            tau = self.baseline_taus[env_type]
            inflated = []
            for obs in perceived:
                inflated.append((
                    max(0, obs[0] - tau),
                    max(0, obs[1] - tau),
                    obs[2] + 2 * tau,
                    obs[3] + 2 * tau
                ))
            
            np.random.seed(seed + 1000)  # Different seed for Standard CP
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
            path = planner.plan()
            results['standard']['times'].append(time.time() - start)
            
            if path:
                results['standard']['paths'] += 1
                results['standard']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['standard']['collisions'] += 1
                        break
            
            # LEARNABLE CP
            start = time.time()
            adaptive_inflated = []
            for obs in perceived:
                cx, cy = obs[0] + obs[2]/2, obs[1] + obs[3]/2
                local_tau = self.predict_tau(cx, cy, perceived, goal=(45, 15))
                results['learnable']['taus'].append(local_tau)
                
                adaptive_inflated.append((
                    max(0, obs[0] - local_tau),
                    max(0, obs[1] - local_tau),
                    obs[2] + 2 * local_tau,
                    obs[3] + 2 * local_tau
                ))
            
            np.random.seed(seed + 2000)  # Different seed for Learnable CP
            planner = RRTStar((5, 15), (45, 15), adaptive_inflated, max_iter=500)
            path = planner.plan()
            results['learnable']['times'].append(time.time() - start)
            
            if path:
                results['learnable']['paths'] += 1
                results['learnable']['lengths'].append(len(path))
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        results['learnable']['collisions'] += 1
                        break
            
            # Progress update
            if (trial + 1) % 100 == 0:
                print(f"  Completed {trial + 1}/{num_trials} trials...")
        
        # Print comprehensive results
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        print(f"\n{'Method':<15} {'Paths Found':<15} {'Collisions':<15} "
              f"{'Collision Rate':<15} {'Avg Length':<12} {'Avg Time(s)':<12} {'Avg τ':<10}")
        print("-"*100)
        
        for method, data in results.items():
            if data['paths'] > 0:
                collision_rate = data['collisions'] / data['paths'] * 100
                avg_length = np.mean(data['lengths'])
                avg_time = np.mean(data['times'])
                
                if method == 'learnable':
                    avg_tau = np.mean(data['taus'])
                elif method == 'standard':
                    avg_tau = np.mean(list(self.baseline_taus.values()))
                else:
                    avg_tau = 0
                
                # Wilson confidence interval for collision rate
                ci_low, ci_high = self.wilson_ci(data['collisions'], data['paths'])
                
                print(f"{method.capitalize():<15} {data['paths']:<15} "
                      f"{data['collisions']:<15} "
                      f"{collision_rate:.1f}% [{ci_low*100:.1f}-{ci_high*100:.1f}]  "
                      f"{avg_length:<12.1f} {avg_time:<12.3f} {avg_tau:<10.3f}")
        
        # Check if Standard CP shows proper failure rate
        if results['standard']['paths'] > 0:
            standard_collision_rate = results['standard']['collisions'] / results['standard']['paths']
            expected_rate = self.alpha  # 0.1 for 90% coverage
            print(f"\nStandard CP: Expected ~{expected_rate*100:.0f}% collisions, "
                  f"got {standard_collision_rate*100:.1f}%")
            
            if standard_collision_rate < expected_rate * 0.5:
                print("⚠️  Standard CP is too conservative!")
            elif standard_collision_rate > expected_rate * 1.5:
                print("⚠️  Standard CP is not conservative enough!")
            else:
                print("✓ Standard CP calibration looks reasonable")
        
        # Timing summary
        print(f"\nTiming Summary:")
        print(f"  Calibration: {sum(self.calibration_times.values()):.2f}s total")
        print(f"  Training: {self.training_time:.2f}s")
        print(f"  Avg inference: {np.mean(self.inference_times)*1000:.2f}ms")
        
        return results
    
    def wilson_ci(self, successes, total, confidence=0.95):
        """Wilson score confidence interval"""
        if total == 0:
            return 0, 0
        p = successes / total
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
        return max(0, center - margin), min(1, center + margin)


def main():
    """Run final comprehensive evaluation"""
    
    # Initialize model with 90% coverage
    model = FinalLearnableCP(coverage=0.90, max_tau=1.0)
    
    # Calibrate Standard CP with more samples
    model.calibrate_standard_cp(num_samples=500)
    
    # Train model with more epochs
    history = model.train(num_epochs=150)
    
    # Comprehensive evaluation with more trials
    results = model.evaluate_comprehensive(num_trials=500)  # Full evaluation
    
    # Save model
    torch.save({
        'model_state': model.scoring_net.state_dict(),
        'coverage': model.coverage,
        'max_tau': model.max_tau,
        'baseline_taus': model.baseline_taus,
        'training_history': history
    }, 'final_learnable_cp_model.pth')
    
    print("\nModel and results saved!")
    
    return results


if __name__ == "__main__":
    main()