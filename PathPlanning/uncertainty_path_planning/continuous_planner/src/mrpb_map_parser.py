#!/usr/bin/env python3
"""
MRPB Map Parser - Extract obstacles from actual MRPB dataset maps

This parser reads the PGM occupancy grid maps from the MRPB dataset
and converts them to obstacle representations for our path planning framework.

We use the EXACT maps from MRPB with their original dimensions and robot size.
"""

import numpy as np
import cv2
import yaml
import os
from typing import List, Tuple
from scipy import ndimage
import matplotlib.pyplot as plt


class MRPBMapParser:
    """
    Parse MRPB occupancy grid maps and extract obstacles
    """
    
    def __init__(self, map_name: str = "office01add", 
                 mrpb_path: str = "../mrpb_dataset"):
        """
        Initialize MRPB map parser
        
        Args:
            map_name: Name of the map from MRPB dataset
                     Options: 'office01add', 'office02', 'shopping_mall', 
                             'room02', 'maze', 'track', 'narrow_graph'
            mrpb_path: Path to the cloned MRPB repository
        """
        self.map_name = map_name
        self.mrpb_path = mrpb_path
        
        # Path to maps
        self.map_dir = os.path.join(mrpb_path, 'move_base_benchmark', 'maps', map_name)
        self.pgm_file = os.path.join(self.map_dir, 'map.pgm')
        self.yaml_file = os.path.join(self.map_dir, 'map.yaml')
        
        # Load map metadata
        self.load_map_metadata()
        
        # Load occupancy grid
        self.occupancy_grid = self.load_pgm_map()
        
        # Extract obstacles as rectangles
        self.obstacles = self.extract_obstacles()
        
        # MRPB robot specifications
        self.robot_radius = 0.17  # meters (from MRPB paper)
        
    def load_map_metadata(self):
        """Load map metadata from YAML file"""
        with open(self.yaml_file, 'r') as f:
            metadata = yaml.safe_load(f)
        
        self.resolution = metadata['resolution']  # meters per pixel
        self.origin = metadata['origin']  # [x, y, theta] in meters
        self.occupied_thresh = metadata['occupied_thresh']
        self.free_thresh = metadata['free_thresh']
        
        print(f"Map: {self.map_name}")
        print(f"  Resolution: {self.resolution} m/pixel")
        print(f"  Origin: {self.origin}")
        
    def load_pgm_map(self):
        """Load PGM occupancy grid map"""
        # Read PGM file
        img = cv2.imread(self.pgm_file, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Could not load map file: {self.pgm_file}")
        
        # Convert to occupancy grid
        # In ROS maps: 0 = free, 100 = occupied, 205 = unknown
        # PGM: 0 = black (occupied), 254 = white (free), 205 = gray (unknown)
        
        # Flip vertically (PGM origin is top-left, we want bottom-left)
        img = np.flipud(img)
        
        # Create occupancy grid
        occupancy = np.zeros_like(img, dtype=np.uint8)
        occupancy[img < 128] = 100  # Occupied
        occupancy[img >= 250] = 0   # Free
        occupancy[(img >= 128) & (img < 250)] = 50  # Unknown
        
        # Calculate world dimensions
        self.height_pixels = img.shape[0]
        self.width_pixels = img.shape[1]
        self.width_meters = self.width_pixels * self.resolution
        self.height_meters = self.height_pixels * self.resolution
        
        print(f"  Grid size: {self.width_pixels}x{self.height_pixels} pixels")
        print(f"  World size: {self.width_meters:.1f}x{self.height_meters:.1f} meters")
        
        return occupancy
    
    def extract_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """
        Extract obstacles as rectangles from occupancy grid
        
        Returns:
            List of obstacles as (x, y, width, height) in meters
        """
        obstacles = []
        
        # Create binary mask of occupied cells
        occupied = (self.occupancy_grid == 100).astype(np.uint8)
        
        # Dilate slightly to merge nearby obstacles
        kernel = np.ones((3, 3), np.uint8)
        occupied = cv2.dilate(occupied, kernel, iterations=1)
        
        # Find contours (obstacle boundaries)
        contours, _ = cv2.findContours(occupied, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to bounding rectangles
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small obstacles (noise)
            if w < 3 or h < 3:
                continue
            
            # Convert from pixels to meters
            x_meters = x * self.resolution + self.origin[0]
            y_meters = y * self.resolution + self.origin[1]
            w_meters = w * self.resolution
            h_meters = h * self.resolution
            
            obstacles.append((x_meters, y_meters, w_meters, h_meters))
        
        print(f"  Extracted {len(obstacles)} obstacles")
        
        # Simplify by merging overlapping rectangles
        obstacles = self.merge_overlapping_obstacles(obstacles)
        print(f"  After merging: {len(obstacles)} obstacles")
        
        return obstacles
    
    def merge_overlapping_obstacles(self, obstacles: List[Tuple[float, float, float, float]], 
                                   threshold: float = 0.1) -> List[Tuple[float, float, float, float]]:
        """
        Merge overlapping or nearby obstacles
        
        Args:
            obstacles: List of (x, y, width, height) tuples
            threshold: Distance threshold for merging (meters)
        
        Returns:
            Merged list of obstacles
        """
        if not obstacles:
            return obstacles
        
        merged = []
        used = [False] * len(obstacles)
        
        for i, obs1 in enumerate(obstacles):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = obs1
            
            # Find all overlapping obstacles
            group = [obs1]
            used[i] = True
            
            for j, obs2 in enumerate(obstacles):
                if i == j or used[j]:
                    continue
                
                x2, y2, w2, h2 = obs2
                
                # Check if rectangles overlap or are very close
                if (abs(x1 - x2) < w1 + w2 + threshold and 
                    abs(y1 - y2) < h1 + h2 + threshold):
                    group.append(obs2)
                    used[j] = True
            
            # Merge the group into one rectangle
            if group:
                min_x = min(obs[0] for obs in group)
                min_y = min(obs[1] for obs in group)
                max_x = max(obs[0] + obs[2] for obs in group)
                max_y = max(obs[1] + obs[3] for obs in group)
                
                merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
        
        return merged
    
    def get_free_space_points(self, num_points: int = 100) -> List[Tuple[float, float]]:
        """
        Get random points in free space for start/goal positions
        
        Args:
            num_points: Number of free points to sample
        
        Returns:
            List of (x, y) points in meters
        """
        free_points = []
        
        # Find all free pixels
        free_pixels = np.argwhere(self.occupancy_grid == 0)
        
        if len(free_pixels) < num_points:
            num_points = len(free_pixels)
        
        # Random sample
        indices = np.random.choice(len(free_pixels), num_points, replace=False)
        
        for idx in indices:
            y_pixel, x_pixel = free_pixels[idx]
            
            # Convert to meters
            x_meters = x_pixel * self.resolution + self.origin[0]
            y_meters = y_pixel * self.resolution + self.origin[1]
            
            free_points.append((x_meters, y_meters))
        
        return free_points
    
    def visualize_map(self, show_grid: bool = False):
        """
        Visualize the parsed map with obstacles
        
        Args:
            show_grid: If True, show occupancy grid alongside obstacles
        """
        if show_grid:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Show occupancy grid
            ax1.imshow(self.occupancy_grid, cmap='gray', origin='lower')
            ax1.set_title(f'MRPB {self.map_name}: Occupancy Grid')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            
            ax = ax2
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw obstacles
        for obs in self.obstacles:
            x, y, w, h = obs
            rect = plt.Rectangle((x, y), w, h, 
                                facecolor='gray', alpha=0.7, 
                                edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        
        # Draw robot size reference
        robot_circle = plt.Circle((0, 0), self.robot_radius, 
                                 color='green', fill=False, 
                                 linewidth=2, label=f'Robot (r={self.robot_radius}m)')
        ax.add_patch(robot_circle)
        
        # Set limits
        ax.set_xlim(self.origin[0], self.origin[0] + self.width_meters)
        ax.set_ylim(self.origin[1], self.origin[1] + self.height_meters)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'MRPB {self.map_name}: Extracted Obstacles')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'mrpb_{self.map_name}_parsed.png', dpi=150)
        plt.show()
    
    def save_obstacles(self, filename: str = None):
        """
        Save obstacles to a file for later use
        
        Args:
            filename: Output filename (default: map_name_obstacles.npy)
        """
        if filename is None:
            filename = f"{self.map_name}_obstacles.npy"
        
        np.save(filename, self.obstacles)
        print(f"Saved obstacles to {filename}")
    
    def get_environment_info(self) -> dict:
        """
        Get complete environment information
        
        Returns:
            Dictionary with all environment parameters
        """
        return {
            'map_name': self.map_name,
            'width_meters': self.width_meters,
            'height_meters': self.height_meters,
            'resolution': self.resolution,
            'origin': self.origin,
            'robot_radius': self.robot_radius,
            'num_obstacles': len(self.obstacles),
            'obstacles': self.obstacles
        }


def parse_all_mrpb_maps(mrpb_path: str = "../mrpb_dataset"):
    """
    Parse all available MRPB maps
    
    Args:
        mrpb_path: Path to MRPB repository
    
    Returns:
        Dictionary of parsed environments
    """
    # Available maps in MRPB dataset
    map_names = [
        'office01add',      # Indoor office
        'office02',         # Larger office
        'shopping_mall',    # Mall environment
        'room02',          # Family house
        'maze',            # Challenging maze
        'track',           # U-shaped track
        'narrow_graph'     # Acute angles
    ]
    
    environments = {}
    
    for map_name in map_names:
        try:
            print(f"\nParsing {map_name}...")
            parser = MRPBMapParser(map_name, mrpb_path)
            environments[map_name] = parser.get_environment_info()
            
            # Visualize
            parser.visualize_map(show_grid=False)
            
        except Exception as e:
            print(f"Error parsing {map_name}: {e}")
    
    return environments


if __name__ == "__main__":
    # Test with single map
    print("Testing MRPB Map Parser with office01add...")
    
    try:
        parser = MRPBMapParser('office01add')
        parser.visualize_map(show_grid=True)
        
        # Get some free points for testing
        free_points = parser.get_free_space_points(5)
        print(f"\nSample free space points for start/goal:")
        for i, point in enumerate(free_points):
            print(f"  Point {i+1}: ({point[0]:.2f}, {point[1]:.2f}) meters")
        
        # Save obstacles
        parser.save_obstacles()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the MRPB dataset is cloned in ../mrpb_dataset/")
        print("Run: git clone https://github.com/NKU-MobFly-Robotics/local-planning-benchmark.git mrpb_dataset")