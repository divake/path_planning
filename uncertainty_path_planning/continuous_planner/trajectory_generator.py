#!/usr/bin/env python3
"""
Trajectory Generation Module for Path Planning

This module provides utilities for generating realistic robot trajectories
from waypoints using quintic polynomial interpolation.
"""

import numpy as np
import sys
from typing import List, Tuple, Optional
from mrpb_metrics import NavigationData, calculate_obstacle_distance

# Add path for CurvesGenerator
sys.path.append('/mnt/ssd1/divake/path_planning/CurvesGenerator')
from quintic_polynomial import QuinticPolynomial


def generate_realistic_trajectory(waypoints: List[Tuple[float, float]], 
                                 parser,
                                 planning_time: Optional[float] = None,
                                 max_vel: float = 1.5,
                                 max_acc: float = 2.0) -> List[NavigationData]:
    """
    Generate realistic robot trajectory from RRT* waypoints using QuinticPolynomial
    
    This function takes discrete waypoints from a path planner and generates
    a smooth, continuous trajectory with realistic velocity profiles.
    
    Args:
        waypoints: List of [x, y] waypoint coordinates from path planner
        parser: MRPBMapParser object for obstacle distance calculation
        planning_time: Total RRT* planning time to distribute across trajectory
        max_vel: Maximum robot velocity in m/s (default: 1.5)
        max_acc: Maximum robot acceleration in m/s² (default: 2.0)
    
    Returns:
        List of NavigationData objects with position, velocity, and metrics
        
    Note:
        The trajectory is sampled at 50Hz (0.02s intervals) which is typical
        for robot control systems.
    """
    if len(waypoints) < 2:
        return []
    
    trajectory_data = []
    current_time = 0.0
    
    # Process each segment between consecutive waypoints
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        
        # Calculate segment parameters
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        angle = np.arctan2(y1 - y0, x1 - x0)
        
        # Initial and final velocities
        # Start and stop at 0 velocity, continuous velocity between segments
        v0 = 0.0 if i == 0 else min(max_vel, distance / 0.5)
        v1 = 0.0 if i == len(waypoints) - 2 else min(max_vel, distance / 0.5)
        
        # Initial and final accelerations (zero for smooth motion)
        a0 = 0.0
        a1 = 0.0
        
        # Time for this segment (based on average velocity)
        avg_vel = (v0 + v1) / 2 if (v0 + v1) > 0 else max_vel / 2
        T = max(distance / avg_vel, 0.5)  # Minimum 0.5s per segment
        
        # Generate quintic polynomial trajectories for x and y
        traj_x = QuinticPolynomial(x0, v0 * np.cos(angle), a0, 
                                  x1, v1 * np.cos(angle), a1, T)
        traj_y = QuinticPolynomial(y0, v0 * np.sin(angle), a0,
                                  y1, v1 * np.sin(angle), a1, T)
        
        # Sample trajectory at realistic robot control frequency (50Hz = 0.02s)
        dt = 0.02
        num_steps = max(int(T / dt), 5)  # At least 5 points per segment
        
        for step in range(num_steps):
            t = step * T / num_steps
            timestamp = current_time + t
            
            # Calculate position from quintic polynomial
            x = traj_x.calc_xt(t)
            y = traj_y.calc_xt(t)
            
            # Calculate velocity
            vx = traj_x.calc_dxt(t)
            vy = traj_y.calc_dxt(t)
            v = np.sqrt(vx**2 + vy**2)
            
            # Calculate obstacle distance using the utility function
            obs_dist = calculate_obstacle_distance(
                [x, y], 
                parser.occupancy_grid,
                parser.origin, 
                parser.resolution
            )
            
            # Calculate actual computation time per waypoint
            if planning_time is not None and len(waypoints) > 0:
                # Distribute planning time across trajectory points
                total_trajectory_points = (len(waypoints) - 1) * num_steps
                time_cost = planning_time / total_trajectory_points if total_trajectory_points > 0 else 0.01
            else:
                time_cost = 0.01  # Default fallback
            
            # Create navigation data point
            nav_data = NavigationData(
                timestamp=timestamp,
                x=x,
                y=y,
                theta=np.arctan2(vy, vx),
                v=v,
                omega=0.0,  # Angular velocity - simplified for now
                obs_dist=obs_dist,
                time_cost=time_cost  # Actual computation time per step
            )
            
            trajectory_data.append(nav_data)
        
        current_time += T
    
    return trajectory_data


def generate_simple_trajectory(waypoints: List[Tuple[float, float]],
                              parser,
                              dt: float = 0.1) -> List[NavigationData]:
    """
    Generate a simple trajectory by linearly interpolating between waypoints
    
    This is a simpler alternative to generate_realistic_trajectory that
    doesn't use quintic polynomials. Useful for quick testing.
    
    Args:
        waypoints: List of [x, y] waypoint coordinates
        parser: MRPBMapParser object for obstacle distance calculation
        dt: Time step between trajectory points (default: 0.1s)
        
    Returns:
        List of NavigationData objects
    """
    if len(waypoints) < 2:
        return []
    
    trajectory_data = []
    timestamp = 0.0
    
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        num_steps = max(int(distance / 0.1), 2)  # At least 2 points per segment
        
        for step in range(num_steps):
            # Linear interpolation
            alpha = step / num_steps
            x = x0 + alpha * (x1 - x0)
            y = y0 + alpha * (y1 - y0)
            
            # Simple velocity calculation
            if i > 0 or step > 0:
                v = distance / (num_steps * dt)
            else:
                v = 0.0
            
            # Calculate obstacle distance
            obs_dist = calculate_obstacle_distance(
                [x, y],
                parser.occupancy_grid,
                parser.origin,
                parser.resolution
            )
            
            nav_data = NavigationData(
                timestamp=timestamp,
                x=x,
                y=y,
                theta=np.arctan2(y1 - y0, x1 - x0),
                v=v,
                omega=0.0,
                obs_dist=obs_dist,
                time_cost=dt
            )
            
            trajectory_data.append(nav_data)
            timestamp += dt
    
    return trajectory_data