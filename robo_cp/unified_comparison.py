#!/usr/bin/env python3
"""
Unified Comparison Framework for Three Methods:
1. Naive: Direct path planning without uncertainty consideration
2. Standard CP: Fixed tau from calibration dataset  
3. Learnable CP: Adaptive uncertainty prediction

This framework provides fair comparison with proper evaluation metrics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))

import numpy as np
import python_motion_planning as pmp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


class NaiveMethod:
    """Naive path planning without uncertainty consideration."""
    
    def __init__(self, robot_radius: float = 0.5):
        self.robot_radius = robot_radius
        self.name = "Naive"
    
    def plan_path(self, start: Tuple, goal: Tuple, 
                  perceived_obstacles: set) -> Optional[List]:
        """
        Plan path using only perceived obstacles.
        No safety margin or uncertainty handling.
        """
        # Create environment with perceived obstacles
        env = pmp.Grid(51, 31)
        env.update(perceived_obstacles)
        
        # Plan using A*
        factory = pmp.SearchFactory()
        try:
            planner = factory('a_star', start=start, goal=goal, env=env)
            cost, path, _ = planner.plan()
            return path
        except:
            return None
    
    def evaluate_path(self, path: List, true_obstacles: set) -> Dict:
        """Evaluate path against true obstacles."""
        if not path:
            return {'success': False, 'collision': True, 'risk': 1.0}
        
        collisions = 0
        total_risk = 0
        min_clearance = float('inf')
        
        for point in path:
            # Check collision
            min_dist = float('inf')
            for obs in true_obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < self.robot_radius:
                collisions += 1
            
            min_clearance = min(min_clearance, min_dist)
            
            # Compute risk
            if min_dist < self.robot_radius:
                risk = 1.0
            elif min_dist < 2 * self.robot_radius:
                risk = 1.0 - (min_dist - self.robot_radius) / self.robot_radius
            else:
                risk = 0.1 * np.exp(-min_dist / (5 * self.robot_radius))
            
            total_risk += risk
        
        return {
            'success': collisions == 0,
            'collision': collisions > 0,
            'collision_rate': collisions / len(path),
            'avg_risk': total_risk / len(path),
            'min_clearance': min_clearance,
            'path_length': len(path)
        }


class StandardCPMethod:
    """Standard Conformal Prediction with fixed tau from calibration."""
    
    def __init__(self, robot_radius: float = 0.5):
        self.robot_radius = robot_radius
        self.tau = None  # Will be calibrated
        self.name = "Standard CP"
    
    def calibrate(self, calibration_data: Dict, alpha: float = 0.1):
        """
        Calibrate tau using calibration dataset.
        
        Args:
            calibration_data: Dict with features, labels, metadata
            alpha: Miscoverage rate (1-alpha is coverage guarantee)
        """
        # Compute residuals (perception errors)
        residuals = []
        
        for i, meta in enumerate(calibration_data['metadata']):
            # Use clearance as proxy for perception error
            perceived_clearance = calibration_data['features'][i][3] * 10.0
            true_clearance = calibration_data['features'][i][4] * 10.0
            error = abs(perceived_clearance - true_clearance)
            residuals.append(error)
        
        # Set tau as (1-alpha) quantile
        self.tau = np.quantile(residuals, 1 - alpha)
        
        print(f"Standard CP calibrated: tau = {self.tau:.3f}")
        return self.tau
    
    def plan_path_with_safety(self, start: Tuple, goal: Tuple,
                              perceived_obstacles: set) -> Optional[List]:
        """
        Plan path with safety-aware cost function.
        Uses modified A* that penalizes paths close to obstacles.
        """
        if self.tau is None:
            raise ValueError("Must calibrate before planning")
        
        # Use custom A* with safety cost
        path = self._safety_aware_astar(start, goal, perceived_obstacles, self.tau)
        return path
    
    def _safety_aware_astar(self, start: Tuple, goal: Tuple, 
                           obstacles: set, tau: float) -> Optional[List]:
        """Modified A* with safety cost."""
        import heapq
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0),
                          (1,1), (-1,1), (1,-1), (-1,-1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if (0 <= new_pos[0] < 51 and 
                    0 <= new_pos[1] < 31 and
                    new_pos not in obstacles):
                    neighbors.append(new_pos)
            return neighbors
        
        def clearance(point):
            if not obstacles:
                return 10.0
            return min(np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                      for obs in obstacles)
        
        def path_cost(current, neighbor):
            base_cost = np.sqrt((current[0] - neighbor[0])**2 + 
                               (current[1] - neighbor[1])**2)
            
            clear = clearance(neighbor)
            
            # Penalize based on tau
            if clear < tau:
                penalty = 10.0 * (tau - clear) / tau
            elif clear < 2 * tau:
                penalty = 2.0 * (2*tau - clear) / (2*tau)
            else:
                penalty = 0
            
            return base_cost * (1 + penalty)
        
        # A* search
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + path_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def evaluate_path(self, path: List, true_obstacles: set) -> Dict:
        """Evaluate path with tau safety margin."""
        if not path:
            return {'success': False, 'collision': True, 'risk': 1.0}
        
        collisions = 0
        violations = 0  # Points violating tau margin
        total_risk = 0
        min_clearance = float('inf')
        
        for point in path:
            min_dist = float('inf')
            for obs in true_obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < self.robot_radius:
                collisions += 1
            
            if min_dist < self.tau:
                violations += 1
            
            min_clearance = min(min_clearance, min_dist)
            
            # Risk computation
            if min_dist < self.robot_radius:
                risk = 1.0
            elif min_dist < 2 * self.robot_radius:
                risk = 1.0 - (min_dist - self.robot_radius) / self.robot_radius
            else:
                risk = 0.1 * np.exp(-min_dist / (5 * self.robot_radius))
            
            total_risk += risk
        
        return {
            'success': collisions == 0,
            'collision': collisions > 0,
            'collision_rate': collisions / len(path),
            'violation_rate': violations / len(path),
            'avg_risk': total_risk / len(path),
            'min_clearance': min_clearance,
            'path_length': len(path)
        }


class LearnableCPMethod:
    """Learnable Conformal Prediction with adaptive uncertainty."""
    
    def __init__(self, robot_radius: float = 0.5, model_type: str = 'rf'):
        self.robot_radius = robot_radius
        self.model_type = model_type
        self.model = None
        self.calibration_scores = None
        self.alpha = 0.1
        self.name = "Learnable CP"
    
    def train(self, train_data: Dict):
        """
        Train the uncertainty prediction model.
        
        Args:
            train_data: Dict with features and labels
        """
        X_train = train_data['features']
        y_train = train_data['labels']
        
        # Select model
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        elif self.model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Compute training scores for diagnostics
        train_pred = self.model.predict(X_train)
        train_error = np.abs(y_train - train_pred)
        
        print(f"Learnable CP trained ({self.model_type}):")
        print(f"  Training MAE: {np.mean(train_error):.3f}")
        print(f"  Training RMSE: {np.sqrt(np.mean(train_error**2)):.3f}")
    
    def calibrate(self, calibration_data: Dict):
        """
        Calibrate conformal scores on calibration set.
        
        Args:
            calibration_data: Dict with features and labels
        """
        if self.model is None:
            raise ValueError("Must train model before calibration")
        
        X_calib = calibration_data['features']
        y_calib = calibration_data['labels']
        
        # Get predictions
        predictions = self.model.predict(X_calib)
        
        # Compute nonconformity scores
        self.calibration_scores = np.abs(y_calib - predictions)
        
        # Compute quantile for coverage
        q_level = np.ceil((len(self.calibration_scores) + 1) * (1 - self.alpha)) / len(self.calibration_scores)
        q_level = min(q_level, 1.0)
        
        threshold = np.quantile(self.calibration_scores, q_level)
        
        print(f"Learnable CP calibrated:")
        print(f"  Calibration samples: {len(self.calibration_scores)}")
        print(f"  Threshold: {threshold:.3f}")
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict risk with uncertainty bounds.
        
        Returns:
            (prediction, uncertainty_radius)
        """
        if self.model is None or self.calibration_scores is None:
            raise ValueError("Must train and calibrate before prediction")
        
        # Get point prediction
        prediction = self.model.predict(features.reshape(1, -1))[0]
        
        # Compute prediction interval using calibration scores
        q_level = np.ceil((len(self.calibration_scores) + 1) * (1 - self.alpha)) / len(self.calibration_scores)
        q_level = min(q_level, 1.0)
        uncertainty = np.quantile(self.calibration_scores, q_level)
        
        return prediction, uncertainty
    
    def plan_path_with_adaptive_safety(self, start: Tuple, goal: Tuple,
                                       perceived_obstacles: set,
                                       feature_extractor) -> Optional[List]:
        """
        Plan path with adaptive safety margins based on learned uncertainty.
        
        Args:
            start: Start position
            goal: Goal position  
            perceived_obstacles: Perceived obstacle set
            feature_extractor: Function to extract features for a point
        """
        import heapq
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0),
                          (1,1), (-1,1), (1,-1), (-1,-1)]:
                new_pos = (pos[0] + dx, pos[1] + dy)
                if (0 <= new_pos[0] < 51 and 
                    0 <= new_pos[1] < 31 and
                    new_pos not in perceived_obstacles):
                    neighbors.append(new_pos)
            return neighbors
        
        def path_cost(current, neighbor):
            base_cost = np.sqrt((current[0] - neighbor[0])**2 + 
                               (current[1] - neighbor[1])**2)
            
            # Extract features for this point
            features = feature_extractor(neighbor, perceived_obstacles)
            
            # Predict risk and uncertainty
            risk_pred, uncertainty = self.predict_with_uncertainty(features)
            
            # Adaptive penalty based on predicted risk and uncertainty
            if risk_pred > 0.5:
                # High risk region
                penalty = 5.0 * risk_pred * (1 + uncertainty)
            elif risk_pred > 0.2:
                # Medium risk
                penalty = 2.0 * risk_pred * (1 + 0.5 * uncertainty)
            else:
                # Low risk
                penalty = 0.5 * uncertainty
            
            return base_cost * (1 + penalty)
        
        # A* search with adaptive cost
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + path_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def evaluate_path(self, path: List, true_obstacles: set,
                     feature_extractor) -> Dict:
        """Evaluate path with learned uncertainty."""
        if not path:
            return {'success': False, 'collision': True, 'risk': 1.0}
        
        collisions = 0
        total_risk = 0
        total_predicted_risk = 0
        total_uncertainty = 0
        min_clearance = float('inf')
        
        for point in path:
            # True evaluation
            min_dist = float('inf')
            for obs in true_obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < self.robot_radius:
                collisions += 1
            
            min_clearance = min(min_clearance, min_dist)
            
            # True risk
            if min_dist < self.robot_radius:
                risk = 1.0
            elif min_dist < 2 * self.robot_radius:
                risk = 1.0 - (min_dist - self.robot_radius) / self.robot_radius
            else:
                risk = 0.1 * np.exp(-min_dist / (5 * self.robot_radius))
            
            total_risk += risk
            
            # Predicted risk and uncertainty
            features = feature_extractor(point, true_obstacles)
            pred_risk, uncertainty = self.predict_with_uncertainty(features)
            total_predicted_risk += pred_risk
            total_uncertainty += uncertainty
        
        return {
            'success': collisions == 0,
            'collision': collisions > 0,
            'collision_rate': collisions / len(path),
            'avg_risk': total_risk / len(path),
            'avg_predicted_risk': total_predicted_risk / len(path),
            'avg_uncertainty': total_uncertainty / len(path),
            'min_clearance': min_clearance,
            'path_length': len(path)
        }


class UnifiedComparison:
    """Unified comparison framework for all three methods."""
    
    def __init__(self, dataset_path: str = 'enhanced_data'):
        self.dataset_path = dataset_path
        self.methods = {}
        self.results = {}
        
    def load_dataset(self):
        """Load the dataset."""
        self.data = {}
        
        # Load splits
        for split in ['train', 'calibration', 'validation', 'test']:
            data = np.load(f'{self.dataset_path}/{split}_data.npz')
            self.data[split] = {
                'features': data['features'],
                'labels': data['labels']
            }
        
        # Load metadata
        with open(f'{self.dataset_path}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            for split in self.data:
                self.data[split]['metadata'] = metadata['splits'][split]
            self.trials = metadata['trials']
        
        print(f"Dataset loaded from {self.dataset_path}")
        splits_info = ', '.join([f'{k}={len(v["features"])}' for k, v in self.data.items()])
        print(f"  Splits: {splits_info}")
    
    def setup_methods(self):
        """Initialize and setup all three methods."""
        
        # 1. Naive method
        self.methods['naive'] = NaiveMethod()
        
        # 2. Standard CP
        self.methods['standard_cp'] = StandardCPMethod()
        self.methods['standard_cp'].calibrate(self.data['calibration'])
        
        # 3. Learnable CP
        self.methods['learnable_cp'] = LearnableCPMethod(model_type='rf')
        self.methods['learnable_cp'].train(self.data['train'])
        self.methods['learnable_cp'].calibrate(self.data['calibration'])
        
        print("\nAll methods initialized and calibrated")
    
    def create_feature_extractor(self, perceived_obstacles: set):
        """Create a feature extraction function for a given obstacle set."""
        def extract_features(point: Tuple, obstacles: set) -> np.ndarray:
            features = []
            
            # Basic features
            features.extend([point[0] / 50.0, point[1] / 30.0])
            features.append(0.5)  # Dummy progress
            
            # Clearances
            perc_clear = min(np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                            for obs in perceived_obstacles) if perceived_obstacles else 10.0
            true_clear = min(np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                            for obs in obstacles) if obstacles else 10.0
            
            features.extend([min(perc_clear/10, 1), min(true_clear/10, 1)])
            
            # Density
            density = sum(1 for obs in perceived_obstacles
                         if np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2) < 5)
            features.append(min(density/10, 1))
            
            # Goal distance (dummy)
            features.append(0.5)
            
            # Pad to match feature dimension
            while len(features) < 12:
                features.append(0.0)
            
            return np.array(features[:12], dtype=np.float32)
        
        return extract_features
    
    def evaluate_on_test_scenarios(self, num_scenarios: int = 50):
        """
        Evaluate all methods on test scenarios.
        """
        print(f"\nEvaluating on {num_scenarios} test scenarios...")
        
        # Initialize results storage
        for method_name in self.methods:
            self.results[method_name] = {
                'successes': 0,
                'collisions': 0,
                'total_risk': 0,
                'path_lengths': [],
                'computation_times': [],
                'detailed_results': []
            }
        
        # Create test scenarios
        np.random.seed(123)  # Different seed for test
        
        for scenario_id in range(num_scenarios):
            # Create environment
            env_type = np.random.choice(['walls', 'corridor', 'maze', 'cluttered'])
            true_obstacles = self._create_test_environment(env_type)
            
            # Generate start and goal
            start = (np.random.randint(2, 10), np.random.randint(5, 26))
            goal = (np.random.randint(40, 49), np.random.randint(5, 26))
            
            # Add noise to create perceived obstacles
            noise_level = np.random.choice([0.1, 0.3, 0.5])
            perceived_obstacles = self._add_noise(true_obstacles, noise_level)
            
            # Test each method
            for method_name, method in self.methods.items():
                # Plan path
                if method_name == 'naive':
                    path = method.plan_path(start, goal, perceived_obstacles)
                elif method_name == 'standard_cp':
                    path = method.plan_path_with_safety(start, goal, perceived_obstacles)
                else:  # learnable_cp
                    extractor = self.create_feature_extractor(perceived_obstacles)
                    path = method.plan_path_with_adaptive_safety(
                        start, goal, perceived_obstacles, extractor
                    )
                
                # Evaluate path
                if path:
                    if method_name == 'learnable_cp':
                        eval_result = method.evaluate_path(path, true_obstacles, extractor)
                    else:
                        eval_result = method.evaluate_path(path, true_obstacles)
                    
                    # Store results
                    self.results[method_name]['successes'] += eval_result['success']
                    self.results[method_name]['collisions'] += eval_result['collision']
                    self.results[method_name]['total_risk'] += eval_result['avg_risk']
                    self.results[method_name]['path_lengths'].append(eval_result['path_length'])
                    self.results[method_name]['detailed_results'].append(eval_result)
                else:
                    # Planning failed
                    self.results[method_name]['collisions'] += 1
                    self.results[method_name]['detailed_results'].append({
                        'success': False,
                        'collision': True,
                        'avg_risk': 1.0,
                        'path_length': 0
                    })
            
            if (scenario_id + 1) % 10 == 0:
                print(f"  Completed {scenario_id + 1}/{num_scenarios} scenarios")
        
        # Compute final statistics
        self._compute_statistics(num_scenarios)
    
    def _create_test_environment(self, env_type: str) -> set:
        """Create test environment obstacles."""
        obstacles = set()
        
        if env_type == 'walls':
            for i in range(10, 21):
                obstacles.add((i, 15))
            for i in range(15):
                obstacles.add((20, i))
            for i in range(15, 30):
                obstacles.add((30, i))
            for i in range(16):
                obstacles.add((40, i))
        
        elif env_type == 'corridor':
            for i in range(51):
                obstacles.add((i, 8))
                obstacles.add((i, 22))
            for j in range(12, 18):
                obstacles.discard((15, 8))
                obstacles.discard((15, 22))
                obstacles.discard((35, 8))
                obstacles.discard((35, 22))
        
        elif env_type == 'maze':
            for i in range(10, 41, 10):
                for j in range(5, 26):
                    obstacles.add((i, j))
            for i in range(10, 41, 10):
                for gap in [10, 15, 20]:
                    obstacles.discard((i, gap))
        
        elif env_type == 'cluttered':
            num_clusters = np.random.randint(5, 10)
            for _ in range(num_clusters):
                cx = np.random.randint(5, 46)
                cy = np.random.randint(5, 26)
                radius = np.random.randint(2, 4)
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        if dx*dx + dy*dy <= radius*radius:
                            x, y = cx + dx, cy + dy
                            if 0 <= x <= 50 and 0 <= y <= 30:
                                if np.random.random() < 0.7:
                                    obstacles.add((x, y))
        
        return obstacles
    
    def _add_noise(self, true_obstacles: set, noise_level: float) -> set:
        """Add noise to obstacles."""
        perceived = set()
        
        for obs in true_obstacles:
            if np.random.random() < 0.95:  # 5% miss rate
                noise_x = np.random.normal(0, noise_level)
                noise_y = np.random.normal(0, noise_level)
                new_x = int(np.clip(obs[0] + noise_x, 0, 50))
                new_y = int(np.clip(obs[1] + noise_y, 0, 30))
                perceived.add((new_x, new_y))
        
        # Add false positives
        if np.random.random() < noise_level * 0.1:
            num_false = np.random.randint(1, 3)
            for _ in range(num_false):
                x = np.random.randint(5, 46)
                y = np.random.randint(5, 26)
                perceived.add((x, y))
        
        return perceived
    
    def _compute_statistics(self, num_scenarios: int):
        """Compute final statistics for all methods."""
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        for method_name, results in self.results.items():
            success_rate = results['successes'] / num_scenarios
            collision_rate = results['collisions'] / num_scenarios
            avg_risk = results['total_risk'] / num_scenarios
            avg_path_length = np.mean(results['path_lengths']) if results['path_lengths'] else 0
            
            print(f"\n{method_name.upper()}:")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Collision Rate: {collision_rate:.2%}")
            print(f"  Average Risk: {avg_risk:.3f}")
            print(f"  Avg Path Length: {avg_path_length:.1f}")
    
    def visualize_results(self):
        """Create comparison visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        methods = list(self.results.keys())
        colors = ['red', 'blue', 'green']
        
        # Success rates
        ax = axes[0, 0]
        success_rates = [self.results[m]['successes'] / len(self.results[m]['detailed_results']) 
                        for m in methods]
        ax.bar(methods, success_rates, color=colors)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Collision rates
        ax = axes[0, 1]
        collision_rates = [self.results[m]['collisions'] / len(self.results[m]['detailed_results'])
                          for m in methods]
        ax.bar(methods, collision_rates, color=colors)
        ax.set_ylabel('Collision Rate')
        ax.set_title('Collision Rate Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Average risk
        ax = axes[0, 2]
        avg_risks = [self.results[m]['total_risk'] / len(self.results[m]['detailed_results'])
                    for m in methods]
        ax.bar(methods, avg_risks, color=colors)
        ax.set_ylabel('Average Risk')
        ax.set_title('Average Risk Score')
        ax.grid(True, alpha=0.3)
        
        # Path length distribution
        ax = axes[1, 0]
        for method, color in zip(methods, colors):
            if self.results[method]['path_lengths']:
                ax.hist(self.results[method]['path_lengths'], 
                       bins=20, alpha=0.5, label=method, color=color)
        ax.set_xlabel('Path Length')
        ax.set_ylabel('Count')
        ax.set_title('Path Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Risk distribution
        ax = axes[1, 1]
        for method, color in zip(methods, colors):
            risks = [r['avg_risk'] for r in self.results[method]['detailed_results']]
            ax.hist(risks, bins=20, alpha=0.5, label=method, color=color)
        ax.set_xlabel('Average Risk')
        ax.set_ylabel('Count')
        ax.set_title('Risk Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary comparison
        ax = axes[1, 2]
        metrics = ['Success\nRate', 'Safety\n(1-Risk)', 'Efficiency\n(1/Length)']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            success = success_rates[i]
            safety = 1 - avg_risks[i]
            efficiency = 1 / (np.mean(self.results[method]['path_lengths']) / 50) \
                        if self.results[method]['path_lengths'] else 0
            
            values = [success, safety, efficiency]
            ax.bar(x + i*width, values, width, label=method, color=color)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Overall Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Three-Method Comparison: Naive vs Standard CP vs Learnable CP',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/unified_comparison.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to results/unified_comparison.png")
    
    def save_results(self):
        """Save comparison results."""
        os.makedirs('results', exist_ok=True)
        
        # Prepare results dictionary
        save_dict = {
            'methods': list(self.methods.keys()),
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        with open('results/comparison_results.json', 'w') as f:
            # Convert numpy values to Python types
            json_dict = {}
            for method, res in save_dict['results'].items():
                json_dict[method] = {
                    'successes': int(res['successes']),
                    'collisions': int(res['collisions']),
                    'total_risk': float(res['total_risk']),
                    'avg_path_length': float(np.mean(res['path_lengths'])) if res['path_lengths'] else 0,
                    'num_scenarios': len(res['detailed_results'])
                }
            
            save_dict['summary'] = json_dict
            json.dump(save_dict, f, indent=2)
        
        print("Results saved to results/comparison_results.json")


def main():
    """Run the unified comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified method comparison')
    parser.add_argument('--dataset', type=str, default='enhanced_data',
                       help='Dataset directory')
    parser.add_argument('--scenarios', type=int, default=50,
                       help='Number of test scenarios')
    
    args = parser.parse_args()
    
    # Create comparison framework
    comparison = UnifiedComparison(dataset_path=args.dataset)
    
    # Load dataset
    comparison.load_dataset()
    
    # Setup methods
    comparison.setup_methods()
    
    # Run evaluation
    comparison.evaluate_on_test_scenarios(num_scenarios=args.scenarios)
    
    # Visualize results
    comparison.visualize_results()
    
    # Save results
    comparison.save_results()


if __name__ == "__main__":
    main()