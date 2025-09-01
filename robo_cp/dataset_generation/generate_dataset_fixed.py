#!/usr/bin/env python3
"""
CORRECT Dataset Generation for Fixed Wall Environment
This generates dataset for ONE specific environment only!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp
import json
from datetime import datetime
import pickle

class FixedEnvironmentDatasetGenerator:
    """Generate dataset for the FIXED wall environment only."""
    
    def __init__(self, num_samples=1000, robot_radius=0.5, seed=42):
        """
        Initialize dataset generator for fixed environment.
        
        Args:
            num_samples: Number of different start-goal pairs to try
            robot_radius: Robot collision radius
            seed: Random seed
        """
        self.num_samples = num_samples
        self.robot_radius = robot_radius
        self.seed = seed
        np.random.seed(seed)
        
        # Noise levels to test
        self.noise_levels = [0.1, 0.3, 0.5]
        
        # Dataset storage
        self.features = []
        self.labels = []
        self.metadata = []
        
    def create_fixed_wall_environment(self):
        """
        Create the FIXED wall environment.
        This is THE environment we're calibrating for!
        """
        env = pmp.Grid(51, 31)
        obstacles = env.obstacles
        
        # Fixed wall configuration - NEVER CHANGES
        for i in range(10, 21):
            obstacles.add((i, 15))  # Horizontal wall
        for i in range(15):
            obstacles.add((20, i))   # Vertical wall bottom
        for i in range(15, 30):
            obstacles.add((30, i))   # Vertical wall top
        for i in range(16):
            obstacles.add((40, i))   # Vertical wall bottom right
        
        env.update(obstacles)
        return env
    
    def add_perception_noise(self, base_obstacles, noise_level):
        """
        Add perception noise to simulate sensor errors.
        """
        perceived_obstacles = set()
        
        for obs in base_obstacles:
            # 5% chance to miss an obstacle
            if np.random.random() < 0.05:
                continue
            
            # Add Gaussian noise
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)
            
            new_x = int(np.clip(obs[0] + noise_x, 0, 50))
            new_y = int(np.clip(obs[1] + noise_y, 0, 30))
            perceived_obstacles.add((new_x, new_y))
        
        return perceived_obstacles
    
    def compute_clearance(self, point, obstacles):
        """Compute minimum distance to obstacles."""
        if not obstacles:
            return float('inf')
        
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
            min_dist = min(min_dist, dist)
        return min_dist
    
    def compute_collision_risk(self, point, obstacles):
        """
        Compute collision risk (0 to 1).
        """
        clearance = self.compute_clearance(point, obstacles)
        
        if clearance < self.robot_radius:
            return 1.0  # Collision
        elif clearance < 2 * self.robot_radius:
            return 1.0 - (clearance - self.robot_radius) / self.robot_radius
        else:
            return 0.1 / (1.0 + clearance)
    
    def extract_features(self, point, path, point_idx, perceived_obstacles, true_obstacles):
        """
        Extract features for a point.
        """
        features = []
        
        # 1. Position (normalized)
        features.append(point[0] / 50.0)
        features.append(point[1] / 30.0)
        
        # 2. Progress along path
        features.append(point_idx / max(len(path), 1))
        
        # 3. Clearance to perceived obstacles
        perceived_clearance = self.compute_clearance(point, perceived_obstacles)
        features.append(min(perceived_clearance / 10.0, 1.0))
        
        # 4. Clearance to true obstacles
        true_clearance = self.compute_clearance(point, true_obstacles)
        features.append(min(true_clearance / 10.0, 1.0))
        
        # 5. Distance to goal
        goal = path[-1] if path else (45, 15)
        dist_to_goal = np.sqrt((point[0] - goal[0])**2 + (point[1] - goal[1])**2)
        features.append(dist_to_goal / 50.0)
        
        # 6. Uncertainty estimate
        uncertainty = abs(perceived_clearance - true_clearance)
        features.append(min(uncertainty / 5.0, 1.0))
        
        return np.array(features, dtype=np.float32)
    
    def generate_dataset(self):
        """
        Generate dataset for the FIXED environment with various start-goal pairs.
        """
        print(f"Generating dataset for FIXED wall environment...")
        print(f"  Samples: {self.num_samples}")
        print(f"  Noise levels: {self.noise_levels}")
        
        # Create the FIXED environment
        env = self.create_fixed_wall_environment()
        true_obstacles = env.obstacles
        print(f"  Fixed environment has {len(true_obstacles)} obstacles")
        
        # Generate different start-goal scenarios
        scenarios = []
        for _ in range(self.num_samples):
            # Random start and goal positions
            attempts = 0
            while attempts < 100:
                start = (np.random.randint(5, 15), np.random.randint(5, 26))
                goal = (np.random.randint(35, 46), np.random.randint(5, 26))
                if start not in true_obstacles and goal not in true_obstacles:
                    scenarios.append((start, goal))
                    break
                attempts += 1
        
        print(f"  Generated {len(scenarios)} valid start-goal pairs")
        
        # Process each scenario with different noise levels
        total_samples = 0
        for scenario_id, (start, goal) in enumerate(scenarios):
            if scenario_id % 10 == 0:
                print(f"  Processing scenario {scenario_id + 1}/{len(scenarios)}...")
            
            # Try with different noise levels
            for noise_level in self.noise_levels:
                # Add perception noise
                perceived_obstacles = self.add_perception_noise(true_obstacles, noise_level)
                
                # Plan path with perceived obstacles
                perceived_env = pmp.Grid(51, 31)
                perceived_env.update(perceived_obstacles)
                factory = pmp.SearchFactory()
                
                try:
                    planner = factory('a_star', start=start, goal=goal, env=perceived_env)
                    cost, path, _ = planner.plan()
                except:
                    continue
                
                if not path or len(path) < 3:
                    continue
                
                # Extract features for each point
                for i, point in enumerate(path):
                    features = self.extract_features(
                        point, path, i, perceived_obstacles, true_obstacles
                    )
                    
                    # Compute risk label
                    risk = self.compute_collision_risk(point, true_obstacles)
                    
                    self.features.append(features)
                    self.labels.append(risk)
                    self.metadata.append({
                        'scenario_id': scenario_id,
                        'start': start,
                        'goal': goal,
                        'noise_level': noise_level,
                        'point_idx': i,
                        'path_length': len(path)
                    })
                    total_samples += 1
        
        print(f"\nDataset generation complete!")
        print(f"  Total samples: {total_samples}")
        print(f"  Risk distribution: mean={np.mean(self.labels):.3f}, std={np.std(self.labels):.3f}")
        
        return self.features, self.labels, self.metadata
    
    def save_dataset(self, output_dir='data'):
        """Save the dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to arrays
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
        np.savez(f'{output_dir}/train_data_fixed.npz',
                 features=features[train_idx],
                 labels=labels[train_idx])
        
        np.savez(f'{output_dir}/calibration_data_fixed.npz',
                 features=features[calib_idx],
                 labels=labels[calib_idx])
        
        np.savez(f'{output_dir}/test_data_fixed.npz',
                 features=features[test_idx],
                 labels=labels[test_idx])
        
        # Save metadata
        with open(f'{output_dir}/metadata_fixed.pkl', 'wb') as f:
            pickle.dump({
                'train': [self.metadata[i] for i in train_idx],
                'calibration': [self.metadata[i] for i in calib_idx],
                'test': [self.metadata[i] for i in test_idx]
            }, f)
        
        # Save stats
        stats = {
            'num_samples': n_samples,
            'num_train': n_train,
            'num_calib': n_calib,
            'num_test': len(test_idx),
            'feature_dim': features.shape[1],
            'risk_mean': float(np.mean(labels)),
            'risk_std': float(np.std(labels)),
            'environment': 'fixed_wall_environment'
        }
        
        with open(f'{output_dir}/dataset_stats_fixed.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset saved to {output_dir}/")
        print(f"  Train: {n_train} samples")
        print(f"  Calibration: {n_calib} samples")
        print(f"  Test: {len(test_idx)} samples")


def main():
    """Generate dataset for fixed environment."""
    generator = FixedEnvironmentDatasetGenerator(
        num_samples=20,  # Quick generation with 20 start-goal pairs
        robot_radius=0.5
    )
    features, labels, metadata = generator.generate_dataset()
    generator.save_dataset()


if __name__ == "__main__":
    main()