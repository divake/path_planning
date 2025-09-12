#!/usr/bin/env python3
"""
MRPB Metrics Implementation for Uncertainty-Aware Path Planning

Based on the official MRPB metrics from:
"Local Planning Benchmark for Autonomous Mobile Robots"
NKU Mobile & Flying Robotics Lab

Metrics include:
1. Safety Metrics: Minimum distance to obstacles, time in danger zone
2. Efficiency Metrics: Total travel time, computation time
3. Smoothness Metrics: Path smoothness, velocity smoothness
4. Path Quality: Total path length
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class NavigationData:
    """
    Data logged at each planning step during navigation
    
    Attributes:
        timestamp: Time of the i-th planning call (seconds)
        x, y, theta: Robot pose
        v, omega: Linear and angular velocities
        obs_dist: Distance to closest obstacle (meters)
        time_cost: Time consumption of local planning (seconds)
    """
    timestamp: float
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0
    obs_dist: float = float('inf')
    time_cost: float = 0.0


class MRPBMetrics:
    """
    Calculate MRPB standard metrics for path planning evaluation
    """
    
    def __init__(self, safe_distance: float = 0.3):
        """
        Initialize metrics calculator
        
        Args:
            safe_distance: Preset safe distance to obstacles (meters)
        """
        self.safe_distance = safe_distance
        self.data_log: List[NavigationData] = []
        
    def log_data(self, data: NavigationData):
        """Add navigation data point to log"""
        self.data_log.append(data)
    
    def compute_safety_metrics(self) -> Dict[str, float]:
        """
        Compute safety metrics
        
        Returns:
            Dictionary with:
            - d_0: Minimum distance to closest obstacle (meters)
            - d_avg: Average distance to closest obstacle (meters)
            - p_0: Percentage of time spent in dangerous area (%)
        """
        if not self.data_log:
            return {'d_0': float('inf'), 'd_avg': float('inf'), 'p_0': 0.0}
        
        # Equation (1): Minimum distance to obstacles
        d_0 = min(data.obs_dist for data in self.data_log)
        
        # New metric: Average distance to obstacles
        d_avg = sum(data.obs_dist for data in self.data_log) / len(self.data_log)
        
        # Equation (2): Percentage of time in danger zone
        total_time = self.data_log[-1].timestamp - self.data_log[0].timestamp
        if total_time <= 0:
            return {'d_0': d_0, 'd_avg': d_avg, 'p_0': 0.0}
        
        danger_time = 0.0
        start_danger = None
        
        for i, data in enumerate(self.data_log):
            if data.obs_dist < self.safe_distance:
                if start_danger is None:
                    start_danger = i
            else:
                if start_danger is not None:
                    # End of danger zone
                    danger_time += (self.data_log[i-1].timestamp - 
                                  self.data_log[start_danger].timestamp)
                    start_danger = None
        
        # Check if still in danger at end
        if start_danger is not None:
            danger_time += (self.data_log[-1].timestamp - 
                          self.data_log[start_danger].timestamp)
        
        p_0 = (danger_time / total_time) * 100.0
        
        return {'d_0': d_0, 'd_avg': d_avg, 'p_0': p_0}
    
    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """
        Compute efficiency metrics
        
        Returns:
            Dictionary with:
            - T: Total travel time (seconds)
            - C: Average computation time per planning cycle (milliseconds)
        """
        if not self.data_log:
            return {'T': 0.0, 'C': 0.0}
        
        # Equation (3): Total travel time
        T = self.data_log[-1].timestamp - self.data_log[0].timestamp
        
        # Equation (4): Average computation time
        C = sum(data.time_cost for data in self.data_log) / len(self.data_log)
        C *= 1000  # Convert to milliseconds
        
        return {'T': T, 'C': C}
    
    def compute_smoothness_metrics(self) -> Dict[str, float]:
        """
        Compute smoothness metrics
        
        Returns:
            Dictionary with:
            - f_ps: Path smoothness (m^2)
            - f_vs: Velocity smoothness (m/s^2)
        """
        if len(self.data_log) < 2:
            return {'f_ps': 0.0, 'f_vs': 0.0}
        
        # Equation (5): Path smoothness
        f_ps = 0.0
        for i in range(1, len(self.data_log) - 1):
            # Displacement vectors
            dx_prev = self.data_log[i].x - self.data_log[i-1].x
            dy_prev = self.data_log[i].y - self.data_log[i-1].y
            dx_next = self.data_log[i+1].x - self.data_log[i].x
            dy_next = self.data_log[i+1].y - self.data_log[i].y
            
            # Second derivative approximation
            ddx = dx_next - dx_prev
            ddy = dy_next - dy_prev
            
            f_ps += ddx**2 + ddy**2
        
        if len(self.data_log) > 2:
            f_ps /= (len(self.data_log) - 2)
        
        # Equation (6): Velocity smoothness (average acceleration)
        f_vs = 0.0
        for i in range(len(self.data_log) - 1):
            dt = self.data_log[i+1].timestamp - self.data_log[i].timestamp
            if dt > 0:
                dv = abs(self.data_log[i+1].v - self.data_log[i].v)
                f_vs += dv / dt
        
        if len(self.data_log) > 1:
            f_vs /= (len(self.data_log) - 1)
        
        return {'f_ps': f_ps, 'f_vs': f_vs}
    
    def compute_path_length(self) -> float:
        """
        Compute total path length
        
        Returns:
            Total path length in meters
        """
        if len(self.data_log) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(self.data_log) - 1):
            dx = self.data_log[i+1].x - self.data_log[i].x
            dy = self.data_log[i+1].y - self.data_log[i].y
            length += np.sqrt(dx**2 + dy**2)
        
        return length
    
    def compute_all_metrics(self) -> Dict[str, any]:
        """
        Compute all MRPB metrics
        
        Returns:
            Dictionary containing all metrics
        """
        safety = self.compute_safety_metrics()
        efficiency = self.compute_efficiency_metrics()
        smoothness = self.compute_smoothness_metrics()
        path_length = self.compute_path_length()
        
        return {
            'safety': safety,
            'efficiency': efficiency,
            'smoothness': smoothness,
            'path_length': path_length,
            'num_planning_calls': len(self.data_log)
        }
    
    def print_metrics_summary(self):
        """Print formatted metrics summary"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("MRPB METRICS SUMMARY")
        print("="*60)
        
        print("\n1. SAFETY METRICS:")
        print(f"   - Minimum distance to obstacles (d_0): {metrics['safety']['d_0']:.3f} m")
        print(f"   - Time in danger zone (p_0): {metrics['safety']['p_0']:.1f}%")
        
        print("\n2. EFFICIENCY METRICS:")
        print(f"   - Total travel time (T): {metrics['efficiency']['T']:.2f} s")
        print(f"   - Avg computation time (C): {metrics['efficiency']['C']:.1f} ms")
        
        print("\n3. SMOOTHNESS METRICS:")
        print(f"   - Path smoothness (f_ps): {metrics['smoothness']['f_ps']:.4f} m²")
        print(f"   - Velocity smoothness (f_vs): {metrics['smoothness']['f_vs']:.3f} m/s²")
        
        print("\n4. PATH QUALITY:")
        print(f"   - Total path length: {metrics['path_length']:.2f} m")
        print(f"   - Number of planning calls: {metrics['num_planning_calls']}")
        
        print("="*60)
    
    def reset(self):
        """Reset data log for new navigation task"""
        self.data_log = []


class UncertaintyAwareMetrics(MRPBMetrics):
    """
    Extended metrics for uncertainty-aware planning evaluation
    """
    
    def __init__(self, safe_distance: float = 0.3, alpha: float = 0.1):
        """
        Initialize uncertainty-aware metrics
        
        Args:
            safe_distance: Preset safe distance to obstacles
            alpha: Conformal prediction significance level (1-alpha coverage)
        """
        super().__init__(safe_distance)
        self.alpha = alpha
        self.collision_count = 0
        self.uncertainty_margins = []
        
    def log_uncertainty_margin(self, tau: float):
        """Log uncertainty margin (tau) used at each step"""
        self.uncertainty_margins.append(tau)
    
    def log_collision(self):
        """Log a collision event"""
        self.collision_count += 1
    
    def compute_uncertainty_metrics(self) -> Dict[str, any]:
        """
        Compute metrics specific to uncertainty-aware planning
        
        Returns:
            Dictionary with uncertainty-specific metrics
        """
        if not self.uncertainty_margins:
            return {
                'avg_tau': 0.0,
                'max_tau': 0.0,
                'min_tau': 0.0,
                'tau_variance': 0.0,
                'collision_rate': 0.0
            }
        
        tau_array = np.array(self.uncertainty_margins)
        
        # Collision rate across multiple runs
        total_runs = max(1, len(self.data_log))
        collision_rate = self.collision_count / total_runs
        
        return {
            'avg_tau': np.mean(tau_array),
            'max_tau': np.max(tau_array),
            'min_tau': np.min(tau_array),
            'tau_variance': np.var(tau_array),
            'collision_rate': collision_rate,
            'coverage': 1.0 - collision_rate,  # Empirical coverage
            'expected_coverage': 1.0 - self.alpha  # Theoretical coverage
        }
    
    def compute_all_metrics_with_uncertainty(self) -> Dict[str, any]:
        """Compute all metrics including uncertainty-aware ones"""
        base_metrics = self.compute_all_metrics()
        uncertainty_metrics = self.compute_uncertainty_metrics()
        
        return {
            **base_metrics,
            'uncertainty': uncertainty_metrics
        }
    
    def print_uncertainty_metrics_summary(self):
        """Print extended metrics summary with uncertainty metrics"""
        self.print_metrics_summary()
        
        uncertainty = self.compute_uncertainty_metrics()
        
        print("\n5. UNCERTAINTY-AWARE METRICS:")
        print(f"   - Average tau (safety margin): {uncertainty['avg_tau']:.3f} m")
        print(f"   - Tau range: [{uncertainty['min_tau']:.3f}, {uncertainty['max_tau']:.3f}] m")
        print(f"   - Tau variance: {uncertainty['tau_variance']:.4f}")
        print(f"   - Empirical coverage: {uncertainty['coverage']*100:.1f}%")
        print(f"   - Expected coverage: {uncertainty['expected_coverage']*100:.1f}%")
        print(f"   - Collision rate: {uncertainty['collision_rate']*100:.1f}%")
        print("="*60)


# Example usage for testing
if __name__ == "__main__":
    # Create metrics calculator
    metrics = UncertaintyAwareMetrics(safe_distance=0.3, alpha=0.1)
    
    # Simulate navigation data
    print("Simulating navigation with sample data...")
    t = 0.0
    x, y = 0.0, 0.0
    
    for i in range(10):
        data = NavigationData(
            timestamp=t,
            x=x,
            y=y,
            theta=0.0,
            v=0.5,
            omega=0.0,
            obs_dist=0.5 + 0.1 * i,  # Distance increases
            time_cost=0.05  # 50ms per planning cycle
        )
        metrics.log_data(data)
        metrics.log_uncertainty_margin(0.2 + 0.01 * i)  # Tau varies
        
        # Update position
        t += 0.1
        x += 0.5 * 0.1
        y += 0.1 * 0.1
    
    # Print results
    metrics.print_uncertainty_metrics_summary()