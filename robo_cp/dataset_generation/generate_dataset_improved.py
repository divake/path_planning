#!/usr/bin/env python3
"""
Improved Dataset Generation for Uncertainty-Aware Path Planning
- Proper collision detection with robot radius
- Continuous risk labels instead of binary
- More realistic noise model
- Diverse environments
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp
import json
from datetime import datetime
import pickle

class ImprovedDatasetGenerator:
    """Generate improved dataset for learning uncertainty in path planning."""
    
    def __init__(self, num_samples=1000, robot_radius=0.5, seed=42):
        """
        Initialize dataset generator.
        
        Args:
            num_samples: Total number of trials to generate
            robot_radius: Robot collision radius
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.robot_radius = robot_radius
        self.seed = seed
        np.random.seed(seed)
        
        # Noise levels for balanced distribution
        self.noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # More granular
        
        # Dataset storage
        self.trials = []
        self.features = []
        self.labels = []  # Will be continuous risk scores
        self.metadata = []
        
    def create_diverse_environment(self, env_type='walls'):
        """
        Create diverse environments for better generalization.
        
        Args:
            env_type: Type of environment ('walls', 'corridor', 'maze', 'random')
        """
        env = pmp.Grid(51, 31)
        obstacles = env.obstacles
        
        if env_type == 'walls':
            # Original wall configuration
            for i in range(10, 21):
                obstacles.add((i, 15))
            for i in range(15):
                obstacles.add((20, i))
            for i in range(15, 30):
                obstacles.add((30, i))
            for i in range(16):
                obstacles.add((40, i))
                
        elif env_type == 'corridor':
            # Narrow corridor
            for i in range(51):
                obstacles.add((i, 10))
                obstacles.add((i, 20))
            # Add gaps
            for i in range(15, 20):
                obstacles.discard((i, 10))
                obstacles.discard((i, 20))
            for i in range(30, 35):
                obstacles.discard((i, 10))
                obstacles.discard((i, 20))
                
        elif env_type == 'maze':
            # Simple maze
            for i in range(10, 41, 10):
                for j in range(5, 26):
                    obstacles.add((i, j))
            # Add passages
            for i in range(10, 41, 10):
                obstacles.discard((i, 15))
                obstacles.discard((i, 10))
                obstacles.discard((i, 20))
                
        elif env_type == 'random':
            # Random obstacles
            num_obstacles = np.random.randint(20, 50)
            for _ in range(num_obstacles):
                x = np.random.randint(5, 46)
                y = np.random.randint(5, 26)
                obstacles.add((x, y))
                # Add clusters
                if np.random.random() < 0.3:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if 5 <= x+dx <= 45 and 5 <= y+dy <= 25:
                                obstacles.add((x+dx, y+dy))
        
        env.update(obstacles)
        return env
    
    def add_realistic_noise(self, base_obstacles, noise_level, robot_position=None):
        """
        Add realistic perception noise that depends on distance.
        
        Args:
            base_obstacles: Original obstacle positions
            noise_level: Base noise standard deviation
            robot_position: Current robot position for distance-dependent noise
        """
        perceived_obstacles = set()
        
        if robot_position is None:
            robot_position = (25, 15)  # Default center position
        
        for obs in base_obstacles:
            # Distance-dependent noise
            dist_to_robot = np.sqrt((obs[0] - robot_position[0])**2 + 
                                   (obs[1] - robot_position[1])**2)
            
            # Noise increases with distance
            actual_noise = noise_level * (1 + 0.02 * dist_to_robot)
            
            # Detection probability decreases with distance
            detect_prob = 0.98 * np.exp(-dist_to_robot / 100)
            
            if np.random.random() < detect_prob:
                # Add Gaussian noise
                noise_x = np.random.normal(0, actual_noise)
                noise_y = np.random.normal(0, actual_noise)
                
                new_x = int(np.clip(obs[0] + noise_x, 0, 50))
                new_y = int(np.clip(obs[1] + noise_y, 0, 30))
                perceived_obstacles.add((new_x, new_y))
        
        # Add false positives (phantom obstacles)
        if np.random.random() < 0.05:  # 5% chance
            num_false = np.random.randint(1, 3)
            for _ in range(num_false):
                x = np.random.randint(5, 46)
                y = np.random.randint(5, 26)
                perceived_obstacles.add((x, y))
        
        return perceived_obstacles
    
    def check_collision_with_radius(self, point, obstacles):
        """
        Check if robot collides considering its radius.
        
        Args:
            point: Robot position (x, y)
            obstacles: Set of obstacle positions
        
        Returns:
            bool: True if collision occurs
        """
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            if dist < self.robot_radius:
                return True
        return False
    
    def compute_collision_risk(self, point, obstacles):
        """
        Compute continuous collision risk score.
        
        Args:
            point: Position to evaluate
            obstacles: Set of obstacle positions
        
        Returns:
            float: Risk score in [0, 1]
        """
        if not obstacles:
            return 0.0
        
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            min_dist = min(min_dist, dist)
        
        # Convert distance to risk score
        if min_dist < self.robot_radius:
            return 1.0  # Collision
        elif min_dist < 2 * self.robot_radius:
            # High risk zone
            return 1.0 - (min_dist - self.robot_radius) / self.robot_radius
        elif min_dist < 5 * self.robot_radius:
            # Medium risk zone
            return 0.5 * (1.0 - (min_dist - 2*self.robot_radius) / (3*self.robot_radius))
        else:
            # Low risk
            return 0.1 / (1.0 + min_dist)
    
    def extract_features_fast(self, point, path, point_idx, perceived_obstacles, true_obstacles):
        """Fast version with index passed in."""
        features = []
        
        # 1. Basic position (normalized)
        features.append(point[0] / 50.0)
        features.append(point[1] / 30.0)
        
        # 2. Progress along path (use provided index)
        features.append(point_idx / max(len(path), 1))
        
        # 3. Clearance to perceived obstacles
        perceived_clearance = self.compute_clearance(point, perceived_obstacles)
        features.append(min(perceived_clearance / 10.0, 1.0))
        
        # 4. Clearance to true obstacles
        true_clearance = self.compute_clearance(point, true_obstacles)
        features.append(min(true_clearance / 10.0, 1.0))
        
        # 5. Local obstacle density
        local_density = self.compute_local_density(point, perceived_obstacles, radius=5.0)
        features.append(min(local_density / 10.0, 1.0))
        
        # 6. Distance to goal
        goal = path[-1] if path else (45, 15)
        dist_to_goal = np.sqrt((point[0] - goal[0])**2 + (point[1] - goal[1])**2)
        features.append(dist_to_goal / 50.0)
        
        # 7. Path curvature (simplified)
        features.append(0.0)  # Skip for speed
        
        # 8. Uncertainty estimate
        uncertainty = abs(perceived_clearance - true_clearance)
        features.append(min(uncertainty / 5.0, 1.0))
        
        # 9. Risk gradient (simplified)
        features.append(0.0)  # Skip for speed
        
        return np.array(features, dtype=np.float32)
    
    def extract_features(self, point, path, perceived_obstacles, true_obstacles):
        """
        Extract comprehensive features for a point.
        
        Args:
            point: Current point
            path: Full path
            perceived_obstacles: What robot sees
            true_obstacles: Ground truth obstacles
        """
        features = []
        
        # 1. Basic position (normalized)
        features.append(point[0] / 50.0)
        features.append(point[1] / 30.0)
        
        # 2. Progress along path
        if path:
            point_idx = path.index(point) if point in path else 0
            features.append(point_idx / max(len(path), 1))
        else:
            features.append(0.0)
        
        # 3. Clearance to perceived obstacles
        perceived_clearance = self.compute_clearance(point, perceived_obstacles)
        features.append(min(perceived_clearance / 10.0, 1.0))
        
        # 4. Clearance to true obstacles (for training only)
        true_clearance = self.compute_clearance(point, true_obstacles)
        features.append(min(true_clearance / 10.0, 1.0))
        
        # 5. Local obstacle density (perceived)
        local_density = self.compute_local_density(point, perceived_obstacles, radius=5.0)
        features.append(min(local_density / 10.0, 1.0))
        
        # 6. Distance to goal
        goal = path[-1] if path else (45, 15)
        dist_to_goal = np.sqrt((point[0] - goal[0])**2 + (point[1] - goal[1])**2)
        features.append(dist_to_goal / 50.0)
        
        # 7. Path curvature (local)
        if path and len(path) > 2:
            curvature = self.compute_local_curvature(point, path)
            features.append(min(abs(curvature), 1.0))
        else:
            features.append(0.0)
        
        # 8. Uncertainty estimate (difference between perceived and true)
        uncertainty = abs(perceived_clearance - true_clearance)
        features.append(min(uncertainty / 5.0, 1.0))
        
        # 9. Risk gradient (how risk changes in neighborhood)
        risk_gradient = self.compute_risk_gradient(point, perceived_obstacles)
        features.append(min(risk_gradient, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def compute_clearance(self, point, obstacles):
        """Compute minimum distance to obstacles."""
        if not obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def compute_local_density(self, point, obstacles, radius):
        """Count obstacles within radius."""
        count = 0
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            if dist < radius:
                count += 1
        return count
    
    def compute_local_curvature(self, point, path):
        """Compute path curvature at point."""
        if point not in path:
            return 0.0
        
        idx = path.index(point)
        if idx == 0 or idx == len(path) - 1:
            return 0.0
        
        # Use three points to estimate curvature
        p1 = path[idx - 1]
        p2 = point
        p3 = path[idx + 1]
        
        # Calculate angle change
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        
        return angle / np.pi  # Normalize to [0, 1]
    
    def compute_risk_gradient(self, point, obstacles):
        """Compute how quickly risk changes around point."""
        risks = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (point[0] + dx, point[1] + dy)
                risk = self.compute_collision_risk(neighbor, obstacles)
                risks.append(risk)
        
        if risks:
            return np.std(risks)
        return 0.0
    
    def generate_dataset(self):
        """Generate the complete dataset with improvements."""
        
        print(f"Generating {self.num_samples} samples with improved methodology...", flush=True)
        print(f"Environment types: {['walls', 'corridor', 'maze', 'random']}", flush=True)
        
        env_types = ['walls', 'corridor', 'maze', 'random']
        
        for trial_id in range(self.num_samples):
            print(f"\nTrial {trial_id + 1}/{self.num_samples}...", flush=True)
            # Vary environment type
            env_type = env_types[trial_id % len(env_types)]
            env = self.create_diverse_environment(env_type)
            base_obstacles = env.obstacles
            
            # Random start and goal (with timeout to prevent infinite loop)
            attempts = 0
            while attempts < 100:
                start = (np.random.randint(5, 15), np.random.randint(5, 26))
                goal = (np.random.randint(35, 46), np.random.randint(5, 26))
                if start not in base_obstacles and goal not in base_obstacles:
                    break
                attempts += 1
            
            if attempts >= 100:
                continue  # Skip this trial if can't find valid start/goal
            
            # Random noise level
            noise_level = np.random.choice(self.noise_levels)
            
            # Add realistic noise
            perceived_obstacles = self.add_realistic_noise(
                base_obstacles, noise_level, robot_position=start
            )
            
            # Plan path with A*
            perceived_env = pmp.Grid(51, 31)
            perceived_env.update(perceived_obstacles)
            factory = pmp.SearchFactory()
            planner = factory('a_star', start=start, goal=goal, env=perceived_env)
            
            try:
                cost, path, expand = planner.plan()
            except:
                continue  # Skip if planning fails
            
            if not path or len(path) < 3:
                continue
            
            # Process each point in path
            for i, point in enumerate(path):
                # Extract features (pass index to avoid slow lookup)
                features = self.extract_features_fast(
                    point, path, i, perceived_obstacles, base_obstacles
                )
                
                # Compute continuous risk label
                risk_score = self.compute_collision_risk(point, base_obstacles)
                
                # Store sample
                self.features.append(features)
                self.labels.append(risk_score)
                self.metadata.append({
                    'trial_id': trial_id,
                    'env_type': env_type,
                    'noise_level': noise_level,
                    'point_idx': i,
                    'path_length': len(path),
                    'collision': risk_score > 0.5  # Binary for compatibility
                })
            
            # Print progress less frequently
            if (trial_id + 1) % 10 == 0 or trial_id == 0:
                print(f"  Progress: {trial_id + 1}/{self.num_samples} trials completed")
        
        print(f"Dataset generation complete!")
        print(f"  Total samples: {len(self.features)}")
        print(f"  Risk distribution: mean={np.mean(self.labels):.3f}, std={np.std(self.labels):.3f}")
        
        return self.features, self.labels, self.metadata
    
    def save_dataset(self, output_dir='data'):
        """Save the generated dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy arrays
        features = np.array(self.features, dtype=np.float32)
        labels = np.array(self.labels, dtype=np.float32)
        
        # Split into train/calibration/test
        n_samples = len(features)
        n_train = int(0.6 * n_samples)
        n_calib = int(0.2 * n_samples)
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        calib_idx = indices[n_train:n_train + n_calib]
        test_idx = indices[n_train + n_calib:]
        
        # Save splits
        np.savez(f'{output_dir}/train_data_improved.npz',
                 features=features[train_idx],
                 labels=labels[train_idx])
        
        np.savez(f'{output_dir}/calibration_data_improved.npz',
                 features=features[calib_idx],
                 labels=labels[calib_idx])
        
        np.savez(f'{output_dir}/test_data_improved.npz',
                 features=features[test_idx],
                 labels=labels[test_idx])
        
        # Save metadata
        metadata_splits = {
            'train': [self.metadata[i] for i in train_idx],
            'calibration': [self.metadata[i] for i in calib_idx],
            'test': [self.metadata[i] for i in test_idx]
        }
        
        with open(f'{output_dir}/metadata_improved.pkl', 'wb') as f:
            pickle.dump(metadata_splits, f)
        
        # Save statistics
        stats = {
            'num_samples': n_samples,
            'num_train': n_train,
            'num_calib': n_calib,
            'num_test': len(test_idx),
            'feature_dim': features.shape[1],
            'risk_mean': float(np.mean(labels)),
            'risk_std': float(np.std(labels)),
            'collision_rate': float(np.mean(labels > 0.5)),
            'generation_time': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/dataset_stats_improved.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset saved to {output_dir}/")
        print(f"  Train: {n_train} samples")
        print(f"  Calibration: {n_calib} samples")
        print(f"  Test: {len(test_idx)} samples")


def main():
    """Main function to generate improved dataset."""
    generator = ImprovedDatasetGenerator(num_samples=50, robot_radius=0.5)  # Very small for demo
    features, labels, metadata = generator.generate_dataset()
    generator.save_dataset()


if __name__ == "__main__":
    main()