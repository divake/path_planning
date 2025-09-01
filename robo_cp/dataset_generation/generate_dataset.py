#!/usr/bin/env python3
"""
Dataset Generation Script for Uncertainty-Aware Path Planning
Generates balanced dataset with collision labels for training learnable CP.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp
import json
from datetime import datetime
import pickle

class DatasetGenerator:
    """Generate dataset for learning uncertainty in path planning."""
    
    def __init__(self, num_samples=1000, seed=42):
        """
        Initialize dataset generator.
        
        Args:
            num_samples: Total number of trials to generate
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)
        
        # Noise levels for balanced distribution
        self.noise_levels = [0.1, 0.3, 0.5]  # Low, medium, high
        
        # Dataset storage
        self.trials = []
        self.features = []
        self.labels = []
        self.metadata = []
        
    def create_base_environment(self):
        """Create the fixed wall environment."""
        env = pmp.Grid(51, 31)
        obstacles = env.obstacles
        
        # Add walls (fixed configuration)
        for i in range(10, 21):
            obstacles.add((i, 15))  # Horizontal wall
        for i in range(15):
            obstacles.add((20, i))  # Vertical wall
        for i in range(15, 30):
            obstacles.add((30, i))  # Vertical wall
        for i in range(16):
            obstacles.add((40, i))  # Vertical wall
        
        env.update(obstacles)
        return env
    
    def add_noise_to_obstacles(self, base_obstacles, noise_level):
        """
        Add Gaussian noise to obstacle positions (simulating sensor noise).
        
        Args:
            base_obstacles: Original obstacle positions
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Noisy obstacle set
        """
        noisy_obstacles = set()
        
        for obs in base_obstacles:
            # Add Gaussian noise to position
            noise_x = np.random.normal(0, noise_level)
            noise_y = np.random.normal(0, noise_level)
            
            new_x = obs[0] + noise_x
            new_y = obs[1] + noise_y
            
            # Keep within bounds
            new_x = max(1, min(49, new_x))
            new_y = max(1, min(29, new_y))
            
            noisy_obstacles.add((int(np.round(new_x)), int(np.round(new_y))))
        
        return noisy_obstacles
    
    def extract_features(self, point, path, obstacles, start, goal):
        """
        Extract features at a given point along the path.
        
        Args:
            point: (x, y) current position
            path: Full path
            obstacles: Set of obstacle positions
            start: Start position
            goal: Goal position
            
        Returns:
            10-dimensional feature vector
        """
        x, y = point
        features = []
        
        # 1. Distance to nearest obstacle
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
            min_dist = min(min_dist, dist)
        features.append(min(min_dist / 10.0, 1.0))  # Normalize
        
        # 2. Obstacle density within 5 units
        density_5 = sum(1 for obs in obstacles 
                       if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) <= 5)
        features.append(min(density_5 / 20.0, 1.0))
        
        # 3. Obstacle density within 10 units
        density_10 = sum(1 for obs in obstacles 
                        if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) <= 10)
        features.append(min(density_10 / 40.0, 1.0))
        
        # 4. Passage width estimation
        passage_width = self.estimate_passage_width(point, obstacles)
        features.append(min(passage_width / 10.0, 1.0))
        
        # 5. Distance from start
        dist_from_start = np.sqrt((x - start[0])**2 + (y - start[1])**2)
        features.append(min(dist_from_start / 50.0, 1.0))
        
        # 6. Distance to goal
        dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        features.append(min(dist_to_goal / 50.0, 1.0))
        
        # 7. Path curvature at this point
        curvature = self.calculate_local_curvature(point, path)
        features.append(min(curvature, 1.0))
        
        # 8. Number of escape directions
        escape_dirs = self.count_escape_directions(point, obstacles)
        features.append(escape_dirs / 8.0)
        
        # 9. Clearance variance (how much clearance changes nearby)
        clearance_var = self.calculate_clearance_variance(point, obstacles)
        features.append(min(clearance_var / 5.0, 1.0))
        
        # 10. Is in narrow passage (binary)
        is_narrow = 1.0 if passage_width < 3.0 else 0.0
        features.append(is_narrow)
        
        return np.array(features, dtype=np.float32)
    
    def estimate_passage_width(self, point, obstacles):
        """Estimate width of passage at given point."""
        x, y = point
        
        # Check in 8 directions
        min_width = float('inf')
        for angle in np.linspace(0, np.pi, 4):  # Check 4 perpendicular pairs
            # Check both directions
            clear_dist1 = 10.0
            clear_dist2 = 10.0
            
            for dist in np.linspace(0.5, 10, 20):
                # Direction 1
                x1 = x + dist * np.cos(angle)
                y1 = y + dist * np.sin(angle)
                if (int(x1), int(y1)) in obstacles:
                    clear_dist1 = dist
                    break
                
                # Direction 2 (opposite)
                x2 = x - dist * np.cos(angle)
                y2 = y - dist * np.sin(angle)
                if (int(x2), int(y2)) in obstacles:
                    clear_dist2 = dist
                    break
            
            width = clear_dist1 + clear_dist2
            min_width = min(min_width, width)
        
        return min_width
    
    def calculate_local_curvature(self, point, path):
        """Calculate path curvature near this point."""
        if len(path) < 3:
            return 0.0
        
        # Find closest point in path
        x, y = point
        min_dist = float('inf')
        closest_idx = 0
        
        for i, p in enumerate(path):
            dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Calculate curvature using nearby points
        if 1 <= closest_idx < len(path) - 1:
            p1 = path[closest_idx - 1]
            p2 = path[closest_idx]
            p3 = path[closest_idx + 1]
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Angle change
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                return angle / np.pi  # Normalize
        
        return 0.0
    
    def count_escape_directions(self, point, obstacles):
        """Count number of clear escape directions."""
        x, y = point
        clear_directions = 0
        
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            # Check if direction is clear for at least 3 units
            is_clear = True
            for dist in np.linspace(0.5, 3, 6):
                check_x = x + dist * np.cos(angle)
                check_y = y + dist * np.sin(angle)
                if (int(check_x), int(check_y)) in obstacles:
                    is_clear = False
                    break
            
            if is_clear:
                clear_directions += 1
        
        return clear_directions
    
    def calculate_clearance_variance(self, point, obstacles):
        """Calculate how much clearance varies around this point."""
        x, y = point
        clearances = []
        
        # Sample points in a small radius
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            for r in [1, 2]:
                sample_x = x + r * np.cos(angle)
                sample_y = y + r * np.sin(angle)
                
                # Find nearest obstacle
                min_dist = float('inf')
                for obs in obstacles:
                    dist = np.sqrt((sample_x - obs[0])**2 + (sample_y - obs[1])**2)
                    min_dist = min(min_dist, dist)
                
                clearances.append(min_dist)
        
        return np.std(clearances) if clearances else 0.0
    
    def generate_dataset(self):
        """Generate the complete dataset."""
        print("="*60)
        print("DATASET GENERATION")
        print("="*60)
        print(f"Generating {self.num_samples} samples...")
        
        # Fixed start and goal
        start = (5, 5)
        goal = (45, 25)
        
        # Base environment
        base_env = self.create_base_environment()
        base_obstacles = base_env.obstacles
        
        # Track statistics
        collision_count = 0
        samples_per_noise = {level: 0 for level in self.noise_levels}
        collisions_per_noise = {level: 0 for level in self.noise_levels}
        
        for trial_id in range(self.num_samples):
            if trial_id % 100 == 0:
                print(f"  Progress: {trial_id}/{self.num_samples}")
            
            # Select noise level (balanced)
            noise_idx = trial_id % len(self.noise_levels)
            noise_level = self.noise_levels[noise_idx]
            samples_per_noise[noise_level] += 1
            
            # Add noise to obstacles
            noisy_obstacles = self.add_noise_to_obstacles(base_obstacles, noise_level)
            
            # Create environment with noisy obstacles
            noisy_env = pmp.Grid(51, 31)
            noisy_env.update(noisy_obstacles)
            
            # Plan path on noisy obstacles
            try:
                planner = pmp.AStar(start, goal, noisy_env)
                cost, path, _ = planner.plan()
                
                if path:
                    # Check collision with TRUE obstacles
                    collision = False
                    for p in path:
                        if p in base_obstacles:
                            collision = True
                            collision_count += 1
                            collisions_per_noise[noise_level] += 1
                            break
                    
                    # Sample points along path
                    # Sample more densely near obstacles
                    sample_indices = []
                    for i, p in enumerate(path):
                        # Always include some points
                        if i % 5 == 0:
                            sample_indices.append(i)
                        
                        # Extra samples near obstacles
                        min_dist = min([np.sqrt((p[0]-obs[0])**2 + (p[1]-obs[1])**2) 
                                      for obs in base_obstacles])
                        if min_dist < 5 and np.random.random() < 0.5:
                            sample_indices.append(i)
                    
                    sample_indices = list(set(sample_indices))[:30]  # Max 30 points per path
                    
                    # Extract features for sampled points
                    for idx in sample_indices:
                        point = path[idx]
                        features = self.extract_features(point, path, base_obstacles, start, goal)
                        
                        self.features.append(features)
                        self.labels.append(1.0 if collision else 0.0)
                        self.metadata.append({
                            'trial_id': trial_id,
                            'noise_level': noise_level,
                            'point_idx': idx,
                            'collision': collision
                        })
            
            except Exception as e:
                print(f"    Warning: Trial {trial_id} failed: {e}")
                continue
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Print statistics
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Total samples generated: {len(self.features)}")
        print(f"Total trials with collision: {collision_count}/{self.num_samples} "
              f"({100*collision_count/self.num_samples:.1f}%)")
        print("\nPer noise level:")
        for level in self.noise_levels:
            if samples_per_noise[level] > 0:
                coll_rate = 100 * collisions_per_noise[level] / samples_per_noise[level]
                print(f"  σ={level}: {collisions_per_noise[level]}/{samples_per_noise[level]} "
                      f"collisions ({coll_rate:.1f}%)")
        
        print(f"\nFeature shape: {self.features.shape}")
        print(f"Label shape: {self.labels.shape}")
        print(f"Collision rate in dataset: {100*np.mean(self.labels):.1f}%")
    
    def split_dataset(self, train_ratio=0.6, cal_ratio=0.2):
        """
        Split dataset into train/calibration/test with balanced distribution.
        
        Args:
            train_ratio: Fraction for training
            cal_ratio: Fraction for calibration
        """
        print("\n" + "="*60)
        print("CREATING BALANCED SPLITS")
        print("="*60)
        
        n_samples = len(self.features)
        
        # Group by noise level and collision outcome
        groups = {}
        for i, meta in enumerate(self.metadata):
            key = (meta['noise_level'], meta['collision'])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        
        # Stratified split
        train_indices = []
        cal_indices = []
        test_indices = []
        
        for key, indices in groups.items():
            np.random.shuffle(indices)
            n = len(indices)
            
            n_train = int(n * train_ratio)
            n_cal = int(n * cal_ratio)
            
            train_indices.extend(indices[:n_train])
            cal_indices.extend(indices[n_train:n_train + n_cal])
            test_indices.extend(indices[n_train + n_cal:])
        
        # Create splits
        self.train_features = self.features[train_indices]
        self.train_labels = self.labels[train_indices]
        self.train_metadata = [self.metadata[i] for i in train_indices]
        
        self.cal_features = self.features[cal_indices]
        self.cal_labels = self.labels[cal_indices]
        self.cal_metadata = [self.metadata[i] for i in cal_indices]
        
        self.test_features = self.features[test_indices]
        self.test_labels = self.labels[test_indices]
        self.test_metadata = [self.metadata[i] for i in test_indices]
        
        # Print split statistics
        print(f"Train: {len(self.train_features)} samples, "
              f"{100*np.mean(self.train_labels):.1f}% collision rate")
        print(f"Calibration: {len(self.cal_features)} samples, "
              f"{100*np.mean(self.cal_labels):.1f}% collision rate")
        print(f"Test: {len(self.test_features)} samples, "
              f"{100*np.mean(self.test_labels):.1f}% collision rate")
        
        # Check distribution balance
        self.check_distribution_balance()
    
    def check_distribution_balance(self):
        """Verify no distribution shift between splits."""
        print("\nChecking distribution balance...")
        
        # Check noise level distribution
        for split_name, metadata in [('Train', self.train_metadata),
                                     ('Cal', self.cal_metadata),
                                     ('Test', self.test_metadata)]:
            noise_dist = {}
            for meta in metadata:
                level = meta['noise_level']
                noise_dist[level] = noise_dist.get(level, 0) + 1
            
            total = sum(noise_dist.values())
            print(f"\n{split_name} noise distribution:")
            for level in sorted(noise_dist.keys()):
                print(f"  σ={level}: {100*noise_dist[level]/total:.1f}%")
        
        # Check feature statistics
        print("\nFeature statistics (mean ± std):")
        feature_names = ['Dist to obstacle', 'Density 5m', 'Density 10m', 'Passage width',
                        'Dist from start', 'Dist to goal', 'Curvature', 'Escape dirs',
                        'Clearance var', 'Is narrow']
        
        for i, name in enumerate(feature_names):
            train_mean = np.mean(self.train_features[:, i])
            train_std = np.std(self.train_features[:, i])
            cal_mean = np.mean(self.cal_features[:, i])
            cal_std = np.std(self.cal_features[:, i])
            test_mean = np.mean(self.test_features[:, i])
            test_std = np.std(self.test_features[:, i])
            
            print(f"{name:15s}: Train={train_mean:.3f}±{train_std:.3f}, "
                  f"Cal={cal_mean:.3f}±{cal_std:.3f}, "
                  f"Test={test_mean:.3f}±{test_std:.3f}")
    
    def save_dataset(self, output_dir='data'):
        """Save dataset to files."""
        print("\n" + "="*60)
        print("SAVING DATASET")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train set
        np.savez(os.path.join(output_dir, 'train_data.npz'),
                features=self.train_features,
                labels=self.train_labels)
        
        # Save calibration set
        np.savez(os.path.join(output_dir, 'calibration_data.npz'),
                features=self.cal_features,
                labels=self.cal_labels)
        
        # Save test set
        np.savez(os.path.join(output_dir, 'test_data.npz'),
                features=self.test_features,
                labels=self.test_labels)
        
        # Save metadata
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump({
                'train': self.train_metadata,
                'calibration': self.cal_metadata,
                'test': self.test_metadata
            }, f)
        
        # Save statistics
        stats = {
            'generation_time': datetime.now().isoformat(),
            'num_samples': self.num_samples,
            'total_features': len(self.features),
            'feature_dim': self.features.shape[1],
            'noise_levels': self.noise_levels,
            'splits': {
                'train': {'size': len(self.train_features), 
                         'collision_rate': float(np.mean(self.train_labels))},
                'calibration': {'size': len(self.cal_features),
                              'collision_rate': float(np.mean(self.cal_labels))},
                'test': {'size': len(self.test_features),
                        'collision_rate': float(np.mean(self.test_labels))}
            }
        }
        
        with open(os.path.join(output_dir, 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/")
        print(f"  - train_data.npz")
        print(f"  - calibration_data.npz")
        print(f"  - test_data.npz")
        print(f"  - metadata.pkl")
        print(f"  - dataset_stats.json")


if __name__ == "__main__":
    # Generate dataset
    generator = DatasetGenerator(num_samples=1000, seed=42)
    
    # Generate samples
    generator.generate_dataset()
    
    # Create balanced splits
    generator.split_dataset(train_ratio=0.6, cal_ratio=0.2)
    
    # Save to files
    generator.save_dataset()
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)