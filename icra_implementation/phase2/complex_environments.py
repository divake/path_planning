#!/usr/bin/env python3
"""
Complex environment generators for ICRA-quality demonstrations
These generate obstacles in the format expected by existing Hybrid A* planner
"""

import numpy as np
from typing import Tuple, List

def create_parking_lot() -> Tuple[List[float], List[float]]:
    """
    Create a realistic parking lot scenario
    Returns ox, oy lists as expected by existing planner
    """
    ox, oy = [], []
    
    # Boundary walls
    for i in range(0, 101):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(60.0)
    
    for i in range(0, 61):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(100.0)
        oy.append(float(i))
    
    # Parked cars (two rows)
    car_length = 4.5
    car_width = 2.0
    spacing = 6.0
    
    # Upper row of parked cars
    for i in range(8):
        car_x = 15 + i * spacing
        car_y = 40
        
        # Create car outline with multiple points
        for dx in np.linspace(0, car_length, 10):
            for dy in np.linspace(0, car_width, 5):
                ox.append(car_x + dx)
                oy.append(car_y + dy)
    
    # Lower row of parked cars
    for i in range(8):
        car_x = 15 + i * spacing
        car_y = 15
        
        for dx in np.linspace(0, car_length, 10):
            for dy in np.linspace(0, car_width, 5):
                ox.append(car_x + dx)
                oy.append(car_y + dy)
    
    # Some randomly parked cars (obstacles)
    random_cars = [
        (70, 30, np.pi/6),  # x, y, angle
        (80, 25, -np.pi/4),
        (25, 30, np.pi/3)
    ]
    
    for car_x, car_y, angle in random_cars:
        # Rotated car
        for dx in np.linspace(-car_length/2, car_length/2, 10):
            for dy in np.linspace(-car_width/2, car_width/2, 5):
                # Rotate points
                rx = dx * np.cos(angle) - dy * np.sin(angle) + car_x
                ry = dx * np.sin(angle) + dy * np.cos(angle) + car_y
                ox.append(rx)
                oy.append(ry)
    
    return ox, oy

def create_narrow_corridor() -> Tuple[List[float], List[float]]:
    """
    Create a narrow corridor with varying width
    Tests adaptive uncertainty in tight spaces
    """
    ox, oy = [], []
    
    # Boundaries
    for i in range(0, 101):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(60.0)
    
    for i in range(0, 61):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(100.0)
        oy.append(float(i))
    
    # Create S-shaped corridor
    # First segment - narrow entry
    for y in range(5, 25):
        ox.append(20.0)
        oy.append(float(y))
        ox.append(26.0)  # 6 unit width
        oy.append(float(y))
    
    # Second segment - even narrower
    for x in range(26, 50):
        ox.append(float(x))
        oy.append(25.0)
        ox.append(float(x))
        oy.append(29.0)  # 4 unit width!
    
    # Third segment - widening
    for y in range(29, 45):
        width = 4 + (y - 29) * 0.3  # Gradually widen
        ox.append(50.0)
        oy.append(float(y))
        ox.append(50.0 + width)
        oy.append(float(y))
    
    # Add some obstacles in wider areas
    obstacles = [
        (10, 30, 2),  # x, y, radius
        (70, 20, 3),
        (80, 40, 2.5)
    ]
    
    for obs_x, obs_y, radius in obstacles:
        for angle in np.linspace(0, 2*np.pi, 20):
            ox.append(obs_x + radius * np.cos(angle))
            oy.append(obs_y + radius * np.sin(angle))
    
    return ox, oy

def create_cluttered_warehouse() -> Tuple[List[float], List[float]]:
    """
    Create a cluttered warehouse with shelves and obstacles
    """
    ox, oy = [], []
    
    # Boundaries
    for i in range(0, 121):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(80.0)
    
    for i in range(0, 81):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(120.0)
        oy.append(float(i))
    
    # Shelving units (long rectangles)
    shelves = [
        (20, 20, 30, 3),   # x, y, length, width
        (20, 35, 30, 3),
        (20, 50, 30, 3),
        (60, 20, 30, 3),
        (60, 35, 30, 3),
        (60, 50, 30, 3),
    ]
    
    for shelf_x, shelf_y, length, width in shelves:
        for x in np.linspace(shelf_x, shelf_x + length, 30):
            for y in np.linspace(shelf_y, shelf_y + width, 5):
                ox.append(x)
                oy.append(y)
    
    # Random boxes/obstacles
    np.random.seed(42)
    for _ in range(15):
        box_x = np.random.uniform(5, 115)
        box_y = np.random.uniform(5, 75)
        box_size = np.random.uniform(1, 2.5)
        
        # Skip if too close to shelves
        skip = False
        for shelf_x, shelf_y, length, width in shelves:
            if (shelf_x - 5 < box_x < shelf_x + length + 5 and
                shelf_y - 5 < box_y < shelf_y + width + 5):
                skip = True
                break
        
        if not skip:
            for dx in np.linspace(-box_size/2, box_size/2, 5):
                for dy in np.linspace(-box_size/2, box_size/2, 5):
                    ox.append(box_x + dx)
                    oy.append(box_y + dy)
    
    return ox, oy

def create_maze() -> Tuple[List[float], List[float]]:
    """
    Create a maze-like environment
    """
    ox, oy = [], []
    
    # Boundaries
    for i in range(0, 101):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(60.0)
    
    for i in range(0, 61):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(100.0)
        oy.append(float(i))
    
    # Maze walls
    walls = [
        # Horizontal walls
        (10, 10, 30, 'h'),
        (50, 10, 30, 'h'),
        (20, 20, 25, 'h'),
        (55, 20, 20, 'h'),
        (10, 30, 20, 'h'),
        (40, 30, 30, 'h'),
        (15, 40, 25, 'h'),
        (50, 40, 25, 'h'),
        (20, 50, 30, 'h'),
        (60, 50, 20, 'h'),
        
        # Vertical walls
        (10, 10, 20, 'v'),
        (30, 15, 15, 'v'),
        (45, 10, 20, 'v'),
        (60, 15, 25, 'v'),
        (75, 20, 20, 'v'),
        (20, 30, 20, 'v'),
        (40, 35, 15, 'v'),
        (55, 30, 20, 'v'),
        (70, 30, 20, 'v'),
        (85, 10, 40, 'v'),
    ]
    
    for x, y, length, direction in walls:
        if direction == 'h':
            for i in np.linspace(x, x + length, int(length * 2)):
                ox.append(i)
                oy.append(float(y))
        else:  # vertical
            for i in np.linspace(y, y + length, int(length * 2)):
                ox.append(float(x))
                oy.append(i)
    
    return ox, oy

def create_roundabout() -> Tuple[List[float], List[float]]:
    """
    Create a roundabout scenario
    """
    ox, oy = [], []
    
    # Boundaries
    for i in range(0, 101):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(60.0)
    
    for i in range(0, 61):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(100.0)
        oy.append(float(i))
    
    # Central roundabout
    center_x, center_y = 50, 30
    inner_radius = 8
    outer_radius = 20
    
    # Inner circle (solid obstacle)
    for angle in np.linspace(0, 2*np.pi, 100):
        for r in np.linspace(0, inner_radius, 10):
            ox.append(center_x + r * np.cos(angle))
            oy.append(center_y + r * np.sin(angle))
    
    # Entry/exit roads with barriers
    roads = [
        (0, 30, 30, 5),     # West entry
        (70, 30, 30, 5),    # East entry
        (50, 0, 5, 10),     # South entry
        (50, 50, 5, 10),    # North entry
    ]
    
    # Add barriers along roads
    for road_x, road_y, length, width in roads:
        if length > width:  # Horizontal road
            # Upper barrier
            for x in np.linspace(road_x, road_x + length, int(length)):
                if abs(x - center_x) > outer_radius:
                    ox.append(x)
                    oy.append(road_y - width)
                    ox.append(x)
                    oy.append(road_y + width)
        else:  # Vertical road
            # Side barriers
            for y in np.linspace(road_y, road_y + width, int(width)):
                if abs(y - center_y) > outer_radius:
                    ox.append(road_x - length)
                    oy.append(y)
                    ox.append(road_x + length)
                    oy.append(y)
    
    return ox, oy

def create_dynamic_scenario(scenario_type='random') -> Tuple[List[float], List[float]]:
    """
    Create dynamic/random scenarios for testing
    """
    ox, oy = [], []
    
    # Boundaries
    for i in range(0, 101):
        ox.append(float(i))
        oy.append(0.0)
        ox.append(float(i))
        oy.append(60.0)
    
    for i in range(0, 61):
        ox.append(0.0)
        oy.append(float(i))
        ox.append(100.0)
        oy.append(float(i))
    
    if scenario_type == 'random':
        # Random obstacles of various sizes
        np.random.seed(None)  # Different each time
        n_obstacles = np.random.randint(10, 25)
        
        for _ in range(n_obstacles):
            obs_type = np.random.choice(['circle', 'rectangle', 'line'])
            
            if obs_type == 'circle':
                cx = np.random.uniform(10, 90)
                cy = np.random.uniform(10, 50)
                radius = np.random.uniform(1, 4)
                
                for angle in np.linspace(0, 2*np.pi, 20):
                    ox.append(cx + radius * np.cos(angle))
                    oy.append(cy + radius * np.sin(angle))
            
            elif obs_type == 'rectangle':
                rx = np.random.uniform(10, 85)
                ry = np.random.uniform(10, 45)
                width = np.random.uniform(2, 8)
                height = np.random.uniform(2, 8)
                angle = np.random.uniform(0, np.pi)
                
                for dx in np.linspace(-width/2, width/2, 8):
                    for dy in np.linspace(-height/2, height/2, 8):
                        # Rotate
                        rot_x = dx * np.cos(angle) - dy * np.sin(angle) + rx
                        rot_y = dx * np.sin(angle) + dy * np.cos(angle) + ry
                        ox.append(rot_x)
                        oy.append(rot_y)
            
            else:  # line
                x1 = np.random.uniform(10, 90)
                y1 = np.random.uniform(10, 50)
                length = np.random.uniform(5, 15)
                angle = np.random.uniform(0, 2*np.pi)
                
                x2 = x1 + length * np.cos(angle)
                y2 = y1 + length * np.sin(angle)
                
                for t in np.linspace(0, 1, 20):
                    ox.append(x1 + t * (x2 - x1))
                    oy.append(y1 + t * (y2 - y1))
    
    return ox, oy

def get_all_scenarios() -> dict:
    """Get all available scenarios"""
    return {
        'parking_lot': create_parking_lot(),
        'narrow_corridor': create_narrow_corridor(),
        'cluttered_warehouse': create_cluttered_warehouse(),
        'maze': create_maze(),
        'roundabout': create_roundabout(),
        'random': create_dynamic_scenario('random')
    }

def get_scenario_start_goal(scenario_name: str) -> Tuple[float, float, float, float, float, float]:
    """Get appropriate start and goal positions for each scenario"""
    
    configs = {
        'parking_lot': (5, 30, np.deg2rad(0), 95, 30, np.deg2rad(0)),
        'narrow_corridor': (5, 10, np.deg2rad(0), 65, 40, np.deg2rad(90)),
        'cluttered_warehouse': (5, 5, np.deg2rad(0), 115, 75, np.deg2rad(0)),
        'maze': (5, 5, np.deg2rad(0), 95, 55, np.deg2rad(0)),
        'roundabout': (5, 30, np.deg2rad(0), 95, 30, np.deg2rad(0)),
        'random': (5, 30, np.deg2rad(0), 95, 30, np.deg2rad(0))
    }
    
    return configs.get(scenario_name, (5, 30, 0, 95, 30, 0))