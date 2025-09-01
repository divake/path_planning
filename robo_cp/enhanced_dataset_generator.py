#!/usr/bin/env python3
"""
Enhanced Dataset Generator for Three-Method Comparison
Addresses key issues and provides comprehensive dataset for:
1. Naive method (no uncertainty)
2. Standard CP (fixed tau from calibration)  
3. Learnable CP (adaptive uncertainty)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class EnhancedDatasetGenerator:
    """Enhanced dataset generator with improvements for robust evaluation."""
    
    def __init__(self, 
                 num_trials: int = 1000,
                 samples_per_trial: int = 50,
                 robot_radius: float = 0.5,
                 seed: int = 42):
        """
        Initialize enhanced generator.
        
        Args:
            num_trials: Number of different scenarios to generate
            samples_per_trial: Max samples per trial (path points)
            robot_radius: Robot collision radius
            seed: Random seed
        """
        self.num_trials = num_trials
        self.samples_per_trial = samples_per_trial
        self.robot_radius = robot_radius
        self.seed = seed
        np.random.seed(seed)
        
        # Enhanced noise model with temporal correlation
        self.noise_params = {
            'low': {'mean': 0.1, 'std': 0.05, 'correlation': 0.3},
            'medium': {'mean': 0.3, 'std': 0.1, 'correlation': 0.5},
            'high': {'mean': 0.5, 'std': 0.15, 'correlation': 0.7},
            'extreme': {'mean': 0.8, 'std': 0.2, 'correlation': 0.8}
        }
        
        # Environment configurations
        self.env_configs = {
            'simple': self._create_simple_env,
            'walls': self._create_walls_env,
            'corridor': self._create_corridor_env,
            'maze': self._create_maze_env,
            'cluttered': self._create_cluttered_env,
            'sparse': self._create_sparse_env
        }
        
        # Planner diversity
        self.planners = ['a_star', 'dijkstra']  # Can add more if available
        
        # Storage
        self.data = {
            'features': [],
            'labels': [],
            'metadata': [],
            'trials': []
        }
        
    def _create_simple_env(self) -> pmp.Grid:
        """Simple environment with few obstacles."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        # Add a few simple obstacles
        for i in range(20, 30):
            obstacles.add((i, 15))
        
        grid.update(obstacles)
        return grid
    
    def _create_walls_env(self) -> pmp.Grid:
        """Original wall environment."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        for i in range(10, 21):
            obstacles.add((i, 15))
        for i in range(15):
            obstacles.add((20, i))
        for i in range(15, 30):
            obstacles.add((30, i))
        for i in range(16):
            obstacles.add((40, i))
        
        grid.update(obstacles)
        return grid
    
    def _create_corridor_env(self) -> pmp.Grid:
        """Narrow corridor environment."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        # Create corridor walls
        for i in range(51):
            obstacles.add((i, 8))
            obstacles.add((i, 22))
        
        # Add openings
        for j in range(12, 18):
            obstacles.discard((15, 8))
            obstacles.discard((15, 22))
            obstacles.discard((35, 8))
            obstacles.discard((35, 22))
        
        grid.update(obstacles)
        return grid
    
    def _create_maze_env(self) -> pmp.Grid:
        """Maze-like environment."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        # Create maze structure
        for i in range(10, 41, 10):
            for j in range(5, 26):
                obstacles.add((i, j))
        
        # Add passages
        for i in range(10, 41, 10):
            for gap in [10, 15, 20]:
                obstacles.discard((i, gap))
        
        grid.update(obstacles)
        return grid
    
    def _create_cluttered_env(self) -> pmp.Grid:
        """Heavily cluttered environment."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        # Random clusters of obstacles
        num_clusters = np.random.randint(5, 10)
        for _ in range(num_clusters):
            cx = np.random.randint(5, 46)
            cy = np.random.randint(5, 26)
            radius = np.random.randint(2, 5)
            
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if dx*dx + dy*dy <= radius*radius:
                        x, y = cx + dx, cy + dy
                        if 0 <= x <= 50 and 0 <= y <= 30:
                            if np.random.random() < 0.7:  # Some gaps
                                obstacles.add((x, y))
        
        grid.update(obstacles)
        return grid
    
    def _create_sparse_env(self) -> pmp.Grid:
        """Sparse environment with few scattered obstacles."""
        grid = pmp.Grid(51, 31)
        obstacles = grid.obstacles
        
        # Scattered individual obstacles
        num_obs = np.random.randint(10, 20)
        for _ in range(num_obs):
            x = np.random.randint(5, 46)
            y = np.random.randint(5, 26)
            obstacles.add((x, y))
        
        grid.update(obstacles)
        return grid
    
    def add_correlated_noise(self, 
                            true_obstacles: set,
                            noise_level: str,
                            prev_noise_state: Optional[Dict] = None) -> Tuple[set, Dict]:
        """
        Add temporally correlated noise to obstacles.
        
        Args:
            true_obstacles: Ground truth obstacles
            noise_level: Noise level key
            prev_noise_state: Previous noise state for correlation
            
        Returns:
            Perceived obstacles and new noise state
        """
        params = self.noise_params[noise_level]
        perceived = set()
        new_state = {}
        
        for obs in true_obstacles:
            obs_key = f"{obs[0]}_{obs[1]}"
            
            # Get previous noise or generate new
            if prev_noise_state and obs_key in prev_noise_state:
                prev_noise = prev_noise_state[obs_key]
                # Correlated noise
                correlation = params['correlation']
                new_noise_x = correlation * prev_noise[0] + \
                             (1 - correlation) * np.random.normal(0, params['mean'])
                new_noise_y = correlation * prev_noise[1] + \
                             (1 - correlation) * np.random.normal(0, params['mean'])
            else:
                # New noise
                new_noise_x = np.random.normal(0, params['mean'])
                new_noise_y = np.random.normal(0, params['mean'])
            
            new_state[obs_key] = (new_noise_x, new_noise_y)
            
            # Apply noise
            new_x = int(np.clip(obs[0] + new_noise_x, 0, 50))
            new_y = int(np.clip(obs[1] + new_noise_y, 0, 30))
            
            # Detection probability
            detect_prob = 0.95 - 0.1 * (params['mean'] / 0.8)
            if np.random.random() < detect_prob:
                perceived.add((new_x, new_y))
        
        # Add false positives
        if np.random.random() < params['mean'] * 0.1:
            num_false = np.random.poisson(2)
            for _ in range(num_false):
                x = np.random.randint(5, 46)
                y = np.random.randint(5, 26)
                perceived.add((x, y))
        
        return perceived, new_state
    
    def compute_path_features(self, 
                             point: Tuple[int, int],
                             path: List[Tuple[int, int]],
                             point_idx: int,
                             perceived_obs: set,
                             true_obs: set,
                             goal: Tuple[int, int]) -> np.ndarray:
        """
        Compute comprehensive features for a path point.
        
        Returns:
            Feature vector of size 12
        """
        features = []
        
        # 1-2: Normalized position
        features.extend([point[0] / 50.0, point[1] / 30.0])
        
        # 3: Progress along path
        features.append(point_idx / max(len(path), 1))
        
        # 4-5: Clearances
        perceived_clear = self._min_distance(point, perceived_obs)
        true_clear = self._min_distance(point, true_obs)
        features.extend([
            min(perceived_clear / 10.0, 1.0),
            min(true_clear / 10.0, 1.0)
        ])
        
        # 6: Local density (5-unit radius)
        density = sum(1 for obs in perceived_obs 
                     if self._distance(point, obs) < 5.0)
        features.append(min(density / 10.0, 1.0))
        
        # 7: Distance to goal
        dist_goal = self._distance(point, goal)
        features.append(min(dist_goal / 50.0, 1.0))
        
        # 8: Path curvature (using 3 points if possible)
        if 1 <= point_idx < len(path) - 1:
            curvature = self._compute_curvature(
                path[point_idx-1], point, path[point_idx+1]
            )
        else:
            curvature = 0.0
        features.append(min(abs(curvature), 1.0))
        
        # 9: Uncertainty estimate
        uncertainty = abs(perceived_clear - true_clear)
        features.append(min(uncertainty / 5.0, 1.0))
        
        # 10: Risk gradient (variance in neighborhood)
        risk_grad = self._compute_risk_gradient(point, perceived_obs)
        features.append(min(risk_grad, 1.0))
        
        # 11: Obstacle angle distribution (uniformity)
        angle_var = self._compute_angle_variance(point, perceived_obs)
        features.append(angle_var)
        
        # 12: Path smoothness (if enough points)
        if len(path) > 3:
            smoothness = self._compute_path_smoothness(path, point_idx)
        else:
            smoothness = 1.0
        features.append(smoothness)
        
        return np.array(features, dtype=np.float32)
    
    def _distance(self, p1: Tuple, p2: Tuple) -> float:
        """Euclidean distance between points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _min_distance(self, point: Tuple, obstacles: set) -> float:
        """Minimum distance to any obstacle."""
        if not obstacles:
            return 10.0  # Max clearance
        return min(self._distance(point, obs) for obs in obstacles)
    
    def _compute_curvature(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """Compute curvature using three points."""
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle) / np.pi
    
    def _compute_risk_gradient(self, point: Tuple, obstacles: set) -> float:
        """Compute risk gradient around point."""
        risks = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (point[0] + dx, point[1] + dy)
                risk = 1.0 / (1.0 + self._min_distance(neighbor, obstacles))
                risks.append(risk)
        return np.std(risks) if risks else 0.0
    
    def _compute_angle_variance(self, point: Tuple, obstacles: set, radius: float = 5.0) -> float:
        """Compute variance in obstacle angles (uniformity measure)."""
        angles = []
        for obs in obstacles:
            if self._distance(point, obs) < radius:
                angle = np.arctan2(obs[1] - point[1], obs[0] - point[0])
                angles.append(angle)
        
        if len(angles) < 2:
            return 0.0
        
        # Circular variance
        return 1.0 - np.abs(np.mean(np.exp(1j * np.array(angles))))
    
    def _compute_path_smoothness(self, path: List, idx: int, window: int = 3) -> float:
        """Compute local path smoothness."""
        start = max(0, idx - window)
        end = min(len(path), idx + window + 1)
        
        if end - start < 3:
            return 1.0
        
        segment = path[start:end]
        angles = []
        for i in range(1, len(segment)):
            dx = segment[i][0] - segment[i-1][0]
            dy = segment[i][1] - segment[i-1][1]
            if dx != 0 or dy != 0:
                angles.append(np.arctan2(dy, dx))
        
        if len(angles) < 2:
            return 1.0
        
        angle_changes = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
        return np.exp(-np.mean(angle_changes))
    
    def compute_collision_risk(self, point: Tuple, true_obstacles: set) -> float:
        """
        Compute continuous collision risk score.
        
        Returns:
            Risk score in [0, 1]
        """
        min_dist = self._min_distance(point, true_obstacles)
        
        if min_dist < self.robot_radius:
            return 1.0  # Collision
        elif min_dist < 2 * self.robot_radius:
            # High risk zone
            return 1.0 - (min_dist - self.robot_radius) / self.robot_radius
        elif min_dist < 5 * self.robot_radius:
            # Medium risk zone  
            return 0.5 * (1.0 - (min_dist - 2*self.robot_radius) / (3*self.robot_radius))
        else:
            # Low risk (exponential decay)
            return 0.1 * np.exp(-min_dist / (5 * self.robot_radius))
    
    def generate_trial(self, trial_id: int) -> Dict:
        """
        Generate a single trial with path planning scenario.
        
        Returns:
            Trial data dictionary
        """
        # Select environment type
        env_type = np.random.choice(list(self.env_configs.keys()))
        env = self.env_configs[env_type]()
        true_obstacles = env.obstacles.copy()
        
        # Select noise level
        noise_level = np.random.choice(list(self.noise_params.keys()))
        
        # Generate start and goal
        max_attempts = 100
        for _ in range(max_attempts):
            start = (np.random.randint(2, 10), np.random.randint(5, 26))
            goal = (np.random.randint(40, 49), np.random.randint(5, 26))
            
            if (start not in true_obstacles and 
                goal not in true_obstacles and
                self._min_distance(start, true_obstacles) > self.robot_radius and
                self._min_distance(goal, true_obstacles) > self.robot_radius):
                break
        else:
            return None  # Failed to find valid start/goal
        
        # Generate perceived obstacles with temporal correlation
        perceived_obs, noise_state = self.add_correlated_noise(
            true_obstacles, noise_level, None
        )
        
        # Plan path using perceived obstacles
        perceived_env = pmp.Grid(51, 31)
        perceived_env.update(perceived_obs)
        
        planner_type = np.random.choice(self.planners)
        factory = pmp.SearchFactory()
        
        try:
            planner = factory(planner_type, start=start, goal=goal, env=perceived_env)
            cost, path, expanded = planner.plan()
        except:
            return None  # Planning failed
        
        if not path or len(path) < 5:
            return None  # Path too short
        
        # Collect samples along path
        samples = []
        for i, point in enumerate(path[:self.samples_per_trial]):
            # Update noise correlation for temporal consistency
            if i > 0:
                perceived_obs, noise_state = self.add_correlated_noise(
                    true_obstacles, noise_level, noise_state
                )
            
            # Extract features
            features = self.compute_path_features(
                point, path, i, perceived_obs, true_obstacles, goal
            )
            
            # Compute risk label
            risk = self.compute_collision_risk(point, true_obstacles)
            
            # Check actual collision
            collision = self._min_distance(point, true_obstacles) < self.robot_radius
            
            samples.append({
                'features': features,
                'risk': risk,
                'collision': collision,
                'point': point,
                'clearance': self._min_distance(point, true_obstacles)
            })
        
        return {
            'trial_id': trial_id,
            'env_type': env_type,
            'noise_level': noise_level,
            'planner': planner_type,
            'start': start,
            'goal': goal,
            'path_length': len(path),
            'path_cost': cost,
            'samples': samples,
            'collision_rate': np.mean([s['collision'] for s in samples]),
            'avg_risk': np.mean([s['risk'] for s in samples])
        }
    
    def generate_dataset(self, verbose: bool = True) -> Tuple:
        """
        Generate complete dataset.
        
        Returns:
            features, labels, metadata, trials
        """
        if verbose:
            print(f"Generating enhanced dataset with {self.num_trials} trials...")
            print(f"Environment types: {list(self.env_configs.keys())}")
            print(f"Noise levels: {list(self.noise_params.keys())}")
        
        successful_trials = 0
        trial_id = 0
        
        while successful_trials < self.num_trials:
            if verbose and trial_id % 100 == 0:
                print(f"Progress: {successful_trials}/{self.num_trials} successful trials...")
            
            trial_data = self.generate_trial(trial_id)
            trial_id += 1
            
            if trial_data is None:
                continue
            
            # Store trial
            self.data['trials'].append(trial_data)
            
            # Store samples
            for sample in trial_data['samples']:
                self.data['features'].append(sample['features'])
                self.data['labels'].append(sample['risk'])
                self.data['metadata'].append({
                    'trial_id': trial_data['trial_id'],
                    'env_type': trial_data['env_type'],
                    'noise_level': trial_data['noise_level'],
                    'planner': trial_data['planner'],
                    'collision': sample['collision'],
                    'clearance': sample['clearance']
                })
            
            successful_trials += 1
        
        # Convert to arrays
        features = np.array(self.data['features'], dtype=np.float32)
        labels = np.array(self.data['labels'], dtype=np.float32)
        
        if verbose:
            print(f"\nDataset generation complete!")
            print(f"  Total samples: {len(features)}")
            print(f"  Feature dimension: {features.shape[1]}")
            print(f"  Risk distribution: mean={np.mean(labels):.3f}, std={np.std(labels):.3f}")
            print(f"  Collision rate: {np.mean([m['collision'] for m in self.data['metadata']]):.3f}")
        
        return features, labels, self.data['metadata'], self.data['trials']
    
    def split_dataset(self, features: np.ndarray, labels: np.ndarray, 
                     metadata: List) -> Dict:
        """
        Split dataset for the three methods.
        
        Returns:
            Dictionary with train, calibration, validation, and test splits
        """
        n_samples = len(features)
        indices = np.random.permutation(n_samples)
        
        # Split ratios
        n_train = int(0.5 * n_samples)  # For learnable CP
        n_calib = int(0.2 * n_samples)  # For standard CP and learnable CP
        n_val = int(0.1 * n_samples)    # For hyperparameter tuning
        # Remaining 20% for test
        
        splits = {
            'train': {
                'features': features[indices[:n_train]],
                'labels': labels[indices[:n_train]],
                'metadata': [metadata[i] for i in indices[:n_train]]
            },
            'calibration': {
                'features': features[indices[n_train:n_train+n_calib]],
                'labels': labels[indices[n_train:n_train+n_calib]],
                'metadata': [metadata[i] for i in indices[n_train:n_train+n_calib]]
            },
            'validation': {
                'features': features[indices[n_train+n_calib:n_train+n_calib+n_val]],
                'labels': labels[indices[n_train+n_calib:n_train+n_calib+n_val]],
                'metadata': [metadata[i] for i in indices[n_train+n_calib:n_train+n_calib+n_val]]
            },
            'test': {
                'features': features[indices[n_train+n_calib+n_val:]],
                'labels': labels[indices[n_train+n_calib+n_val:]],
                'metadata': [metadata[i] for i in indices[n_train+n_calib+n_val:]]
            }
        }
        
        return splits
    
    def save_dataset(self, output_dir: str = 'enhanced_data'):
        """Save the generated dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        features, labels, metadata, trials = (
            np.array(self.data['features']),
            np.array(self.data['labels']),
            self.data['metadata'],
            self.data['trials']
        )
        
        # Split dataset
        splits = self.split_dataset(features, labels, metadata)
        
        # Save each split
        for split_name, split_data in splits.items():
            np.savez(f'{output_dir}/{split_name}_data.npz',
                    features=split_data['features'],
                    labels=split_data['labels'])
        
        # Save metadata and trials
        with open(f'{output_dir}/metadata.pkl', 'wb') as f:
            pickle.dump({
                'splits': {k: v['metadata'] for k, v in splits.items()},
                'trials': trials
            }, f)
        
        # Save statistics
        stats = {
            'num_samples': len(features),
            'num_trials': len(trials),
            'splits': {k: len(v['features']) for k, v in splits.items()},
            'feature_dim': features.shape[1],
            'risk_stats': {
                'mean': float(np.mean(labels)),
                'std': float(np.std(labels)),
                'min': float(np.min(labels)),
                'max': float(np.max(labels)),
                'median': float(np.median(labels))
            },
            'collision_rate': float(np.mean([m['collision'] for m in metadata])),
            'env_distribution': {},
            'noise_distribution': {},
            'generation_time': datetime.now().isoformat()
        }
        
        # Count distributions
        for trial in trials:
            env = trial['env_type']
            noise = trial['noise_level']
            stats['env_distribution'][env] = stats['env_distribution'].get(env, 0) + 1
            stats['noise_distribution'][noise] = stats['noise_distribution'].get(noise, 0) + 1
        
        with open(f'{output_dir}/dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset saved to {output_dir}/")
        splits_info = ', '.join([f'{k}={len(v["features"])}' for k, v in splits.items()])
        print(f"  Splits: {splits_info}")
        
        return splits


def main():
    """Generate enhanced dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate enhanced dataset')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--samples', type=int, default=50, help='Max samples per trial')
    parser.add_argument('--output', type=str, default='enhanced_data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generator = EnhancedDatasetGenerator(
        num_trials=args.trials,
        samples_per_trial=args.samples,
        seed=args.seed
    )
    
    # Generate dataset
    features, labels, metadata, trials = generator.generate_dataset(verbose=True)
    
    # Save dataset
    splits = generator.save_dataset(args.output)
    
    # Visualize sample
    print("\nGenerating visualization...")
    visualize_dataset_sample(splits, args.output)


def visualize_dataset_sample(splits: Dict, output_dir: str):
    """Visualize dataset statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Risk distribution per split
    ax = axes[0, 0]
    for split_name in ['train', 'calibration', 'validation', 'test']:
        ax.hist(splits[split_name]['labels'], bins=30, alpha=0.5, label=split_name)
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Count')
    ax.set_title('Risk Distribution by Split')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Collision rate per environment type
    ax = axes[0, 1]
    env_collisions = {}
    for meta in splits['train']['metadata']:
        env = meta['env_type']
        if env not in env_collisions:
            env_collisions[env] = []
        env_collisions[env].append(meta['collision'])
    
    envs = list(env_collisions.keys())
    collision_rates = [np.mean(env_collisions[e]) for e in envs]
    ax.bar(envs, collision_rates)
    ax.set_xlabel('Environment Type')
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Rate by Environment')
    ax.grid(True, alpha=0.3)
    
    # Risk vs clearance
    ax = axes[0, 2]
    clearances = [m['clearance'] for m in splits['train']['metadata']]
    risks = splits['train']['labels']
    ax.scatter(clearances[:1000], risks[:1000], alpha=0.3, s=1)
    ax.set_xlabel('True Clearance')
    ax.set_ylabel('Risk Score')
    ax.set_title('Risk vs Clearance Relationship')
    ax.grid(True, alpha=0.3)
    
    # Feature correlation matrix
    ax = axes[1, 0]
    features = splits['train']['features'][:1000]
    corr = np.corrcoef(features.T)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax)
    
    # Noise level distribution
    ax = axes[1, 1]
    noise_counts = {}
    for meta in splits['train']['metadata']:
        noise = meta['noise_level']
        noise_counts[noise] = noise_counts.get(noise, 0) + 1
    
    ax.bar(noise_counts.keys(), noise_counts.values())
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Sample Count')
    ax.set_title('Noise Level Distribution')
    ax.grid(True, alpha=0.3)
    
    # Feature importance (variance)
    ax = axes[1, 2]
    feature_vars = np.var(splits['train']['features'], axis=0)
    ax.bar(range(len(feature_vars)), feature_vars)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Variance')
    ax.set_title('Feature Variance (Proxy for Importance)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Enhanced Dataset Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_dir}/dataset_visualization.png")


if __name__ == "__main__":
    main()