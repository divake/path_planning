#!/usr/bin/env python3
"""
Ablation Studies Module
Comprehensive experiments for continuous planning
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import random

from rrt_star_planner import RRTStar
from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP, ContinuousNonconformity


class AblationStudies:
    """
    Run ablation studies for continuous planning
    """
    
    def __init__(self, environment: ContinuousEnvironment):
        """
        Initialize ablation studies
        
        Args:
            environment: Continuous environment
        """
        self.env = environment
        self.true_obstacles = environment.obstacles
        
    def study_noise_levels(self, 
                          noise_levels: List[float] = None,
                          num_trials: int = 100) -> Dict:
        """
        Study effect of different noise levels
        
        Args:
            noise_levels: List of noise levels to test
            num_trials: Number of trials per level
            
        Returns:
            Results dictionary
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print("\n" + "="*60)
        print("ABLATION: Noise Level Study")
        print("="*60)
        
        results = {
            'naive': {
                'levels': noise_levels,
                'collision_rates': [],
                'path_lengths': [],
                'success_rates': []
            },
            'standard_cp': {
                'levels': noise_levels,
                'collision_rates': [],
                'path_lengths': [],
                'success_rates': []
            }
        }
        
        for noise_level in noise_levels:
            print(f"\nTesting noise level: {noise_level}")
            
            # Initialize CP for this noise level
            cp = ContinuousStandardCP(self.true_obstacles, "penetration")
            cp.calibrate(ContinuousNoiseModel.add_thinning_noise,
                        {'thin_factor': noise_level},
                        num_samples=100, confidence=0.95)
            
            naive_collisions = 0
            cp_collisions = 0
            naive_lengths = []
            cp_lengths = []
            naive_successes = 0
            cp_successes = 0
            
            for trial in range(num_trials):
                # Generate noisy perception
                perceived = ContinuousNoiseModel.add_thinning_noise(
                    self.true_obstacles, thin_factor=noise_level, seed=trial
                )
                
                # Naive planning
                planner = RRTStar((5, 15), (45, 15), perceived, max_iter=500)
                path = planner.plan()
                
                if path:
                    naive_successes += 1
                    naive_lengths.append(planner.get_metrics()['path_length'])
                    
                    # Check collisions
                    has_collision = any(
                        self.env.point_in_obstacle(p[0], p[1]) for p in path
                    )
                    if has_collision:
                        naive_collisions += 1
                
                # CP planning
                inflated = cp.inflate_obstacles(perceived)
                cp_planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
                cp_path = cp_planner.plan()
                
                if cp_path:
                    cp_successes += 1
                    cp_lengths.append(cp_planner.get_metrics()['path_length'])
                    
                    # Check collisions
                    has_collision = any(
                        self.env.point_in_obstacle(p[0], p[1]) for p in cp_path
                    )
                    if has_collision:
                        cp_collisions += 1
            
            # Store results
            results['naive']['collision_rates'].append(
                (naive_collisions / naive_successes * 100) if naive_successes > 0 else 0
            )
            results['naive']['path_lengths'].append(
                np.mean(naive_lengths) if naive_lengths else 0
            )
            results['naive']['success_rates'].append(
                naive_successes / num_trials * 100
            )
            
            results['standard_cp']['collision_rates'].append(
                (cp_collisions / cp_successes * 100) if cp_successes > 0 else 0
            )
            results['standard_cp']['path_lengths'].append(
                np.mean(cp_lengths) if cp_lengths else 0
            )
            results['standard_cp']['success_rates'].append(
                cp_successes / num_trials * 100
            )
            
            print(f"  Naive: {naive_collisions}/{naive_successes} collisions")
            print(f"  CP: {cp_collisions}/{cp_successes} collisions")
        
        return results
    
    def study_nonconformity_scores(self, num_trials: int = 200) -> Dict:
        """
        Compare different nonconformity score types
        
        Args:
            num_trials: Number of trials
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("ABLATION: Nonconformity Score Comparison")
        print("="*60)
        
        score_types = ["penetration", "hausdorff", "area"]
        results = {}
        
        for score_type in score_types:
            print(f"\nTesting score type: {score_type}")
            
            # Calibrate CP with this score type
            cp = ContinuousStandardCP(self.true_obstacles, score_type)
            tau = cp.calibrate(ContinuousNoiseModel.add_thinning_noise,
                              {'thin_factor': 0.2},
                              num_samples=200, confidence=0.95)
            
            # Get tau curve
            tau_curve = cp.get_tau_curve()
            
            # Test performance
            collisions = 0
            successes = 0
            path_lengths = []
            
            for trial in range(num_trials):
                perceived = ContinuousNoiseModel.add_thinning_noise(
                    self.true_obstacles, thin_factor=0.2, seed=trial+1000
                )
                
                inflated = cp.inflate_obstacles(perceived)
                planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
                path = planner.plan()
                
                if path:
                    successes += 1
                    path_lengths.append(planner.get_metrics()['path_length'])
                    
                    has_collision = any(
                        self.env.point_in_obstacle(p[0], p[1]) for p in path
                    )
                    if has_collision:
                        collisions += 1
            
            results[score_type] = {
                'tau': tau,
                'tau_curve': tau_curve,
                'collision_rate': (collisions / successes * 100) if successes > 0 else 0,
                'success_rate': successes / num_trials * 100,
                'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
                'calibration_scores': cp.calibration_scores
            }
            
            print(f"  τ = {tau:.3f}")
            print(f"  Collision rate: {results[score_type]['collision_rate']:.1f}%")
            print(f"  Success rate: {results[score_type]['success_rate']:.1f}%")
        
        return results
    
    def study_noise_models(self, num_trials: int = 100) -> Dict:
        """
        Compare different noise models
        
        Args:
            num_trials: Number of trials
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("ABLATION: Noise Model Comparison")
        print("="*60)
        
        noise_configs = [
            ("Gaussian σ=0.2", ContinuousNoiseModel.add_gaussian_noise, {'noise_std': 0.2}),
            ("Gaussian σ=0.4", ContinuousNoiseModel.add_gaussian_noise, {'noise_std': 0.4}),
            ("Thinning 20%", ContinuousNoiseModel.add_thinning_noise, {'thin_factor': 0.2}),
            ("Thinning 30%", ContinuousNoiseModel.add_thinning_noise, {'thin_factor': 0.3}),
            ("Expansion 20%", ContinuousNoiseModel.add_expansion_noise, {'expand_factor': 0.2}),
            ("Mixed", ContinuousNoiseModel.add_mixed_noise, 
             {'gaussian_std': 0.1, 'thin_prob': 0.3, 'expand_prob': 0.3})
        ]
        
        results = {}
        
        for name, noise_func, params in noise_configs:
            print(f"\nTesting: {name}")
            
            # Calibrate CP for this noise model
            cp = ContinuousStandardCP(self.true_obstacles, "penetration")
            tau = cp.calibrate(noise_func, params, num_samples=100, confidence=0.95)
            
            # Test naive and CP
            naive_collisions = 0
            cp_collisions = 0
            naive_successes = 0
            cp_successes = 0
            
            for trial in range(num_trials):
                perceived = noise_func(self.true_obstacles, **params, seed=trial)
                
                # Naive
                planner = RRTStar((5, 15), (45, 15), perceived, max_iter=300)
                path = planner.plan()
                
                if path:
                    naive_successes += 1
                    if any(self.env.point_in_obstacle(p[0], p[1]) for p in path):
                        naive_collisions += 1
                
                # CP
                inflated = cp.inflate_obstacles(perceived)
                cp_planner = RRTStar((5, 15), (45, 15), inflated, max_iter=300)
                cp_path = cp_planner.plan()
                
                if cp_path:
                    cp_successes += 1
                    if any(self.env.point_in_obstacle(p[0], p[1]) for p in cp_path):
                        cp_collisions += 1
            
            results[name] = {
                'tau': tau,
                'naive_collision_rate': (naive_collisions / naive_successes * 100) 
                                       if naive_successes > 0 else 0,
                'cp_collision_rate': (cp_collisions / cp_successes * 100) 
                                    if cp_successes > 0 else 0,
                'naive_success_rate': naive_successes / num_trials * 100,
                'cp_success_rate': cp_successes / num_trials * 100
            }
            
            print(f"  τ = {tau:.3f}")
            print(f"  Naive collisions: {results[name]['naive_collision_rate']:.1f}%")
            print(f"  CP collisions: {results[name]['cp_collision_rate']:.1f}%")
        
        return results
    
    def study_computation_time(self, num_trials: int = 50) -> Dict:
        """
        Study computation time for different methods
        
        Args:
            num_trials: Number of trials
            
        Returns:
            Results dictionary
        """
        print("\n" + "="*60)
        print("ABLATION: Computation Time Study")
        print("="*60)
        
        methods = {
            'Naive': None,
            'CP-Uniform': 'uniform',
            'CP-Directional': 'directional'
        }
        
        # Calibrate CP once
        cp = ContinuousStandardCP(self.true_obstacles, "penetration")
        cp.calibrate(ContinuousNoiseModel.add_thinning_noise,
                    {'thin_factor': 0.2},
                    num_samples=100, confidence=0.95)
        
        results = {}
        
        for method_name, inflation_type in methods.items():
            print(f"\nTesting: {method_name}")
            
            times = []
            
            for trial in range(num_trials):
                perceived = ContinuousNoiseModel.add_thinning_noise(
                    self.true_obstacles, thin_factor=0.2, seed=trial
                )
                
                start_time = time.time()
                
                if method_name == 'Naive':
                    obstacles = perceived
                else:
                    obstacles = cp.inflate_obstacles(perceived, inflation_type)
                
                planner = RRTStar((5, 15), (45, 15), obstacles, max_iter=500)
                path = planner.plan()
                
                elapsed = (time.time() - start_time) * 1000  # ms
                times.append(elapsed)
            
            results[method_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
            
            print(f"  Mean time: {results[method_name]['mean_time']:.2f} ms")
            print(f"  Std: {results[method_name]['std_time']:.2f} ms")
        
        return results
    
    def validate_coverage_guarantee(self, 
                                   confidence_levels: List[float] = None,
                                   num_trials: int = 500) -> Dict:
        """
        Validate that CP provides promised coverage
        
        Args:
            confidence_levels: Confidence levels to test
            num_trials: Number of trials
            
        Returns:
            Validation results
        """
        print("\n" + "="*60)
        print("ABLATION: Coverage Guarantee Validation")
        print("="*60)
        
        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95, 0.99]
        
        results = {}
        
        for confidence in confidence_levels:
            print(f"\nTesting {confidence*100:.0f}% confidence level")
            
            # Calibrate CP for this confidence
            cp = ContinuousStandardCP(self.true_obstacles, "penetration")
            tau = cp.calibrate(ContinuousNoiseModel.add_thinning_noise,
                             {'thin_factor': 0.2},
                             num_samples=200, confidence=confidence)
            
            # Test coverage
            covered = 0
            total = 0
            
            for trial in range(num_trials):
                perceived = ContinuousNoiseModel.add_thinning_noise(
                    self.true_obstacles, thin_factor=0.2, seed=trial+5000
                )
                
                inflated = cp.inflate_obstacles(perceived)
                planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
                path = planner.plan()
                
                if path:
                    total += 1
                    has_collision = any(
                        self.env.point_in_obstacle(p[0], p[1]) for p in path
                    )
                    if not has_collision:
                        covered += 1
            
            actual_coverage = (covered / total * 100) if total > 0 else 0
            
            results[f'{confidence*100:.0f}%'] = {
                'target': confidence * 100,
                'actual': actual_coverage,
                'tau': tau,
                'gap': abs(actual_coverage - confidence * 100)
            }
            
            print(f"  Target coverage: {confidence*100:.0f}%")
            print(f"  Actual coverage: {actual_coverage:.1f}%")
            print(f"  τ = {tau:.3f}")
        
        return results