#!/usr/bin/env python3
"""
Create Publication-Ready RRT* Tree Exploration Visualization
Shows the actual RRT* tree search process with nodes and edges
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import yaml
import sys
import os
import random

# Add paths for imports
sys.path.append('..')
sys.path.append('../..')
from mrpb_map_parser import MRPBMapParser

# Modified RRT planner to expose tree structure
class RRTStarGridVisual:
    """RRT* planner with visualization capability"""

    def __init__(self, start, goal, occupancy_grid, origin, resolution,
                 robot_radius=0.17, step_size=2.0, max_iter=2000,
                 goal_threshold=1.0, search_radius=5.0):
        self.start = start
        self.goal = goal
        self.occupancy_grid = occupancy_grid
        self.origin = origin[:2]
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.search_radius = search_radius
        self.goal_sample_rate = 0.2

        # Calculate bounds
        self.height_pixels, self.width_pixels = occupancy_grid.shape
        self.width_meters = self.width_pixels * resolution
        self.height_meters = self.height_pixels * resolution

        self.x_min = self.origin[0]
        self.x_max = self.origin[0] + self.width_meters
        self.y_min = self.origin[1]
        self.y_max = self.origin[1] + self.height_meters

        self.robot_radius_pixels = int(np.ceil(robot_radius / resolution))

        # RRT* tree structure - exposed for visualization
        self.nodes = [start]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        self.edges = []  # Store edges for visualization

    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def is_collision_free(self, x, y):
        grid_x, grid_y = self.world_to_grid(x, y)

        if (grid_x - self.robot_radius_pixels < 0 or
            grid_x + self.robot_radius_pixels >= self.width_pixels or
            grid_y - self.robot_radius_pixels < 0 or
            grid_y + self.robot_radius_pixels >= self.height_pixels):
            return False

        for dx in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
            for dy in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                if dx*dx + dy*dy <= self.robot_radius_pixels * self.robot_radius_pixels:
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    if self.occupancy_grid[check_y, check_x] > 50:
                        return False
        return True

    def is_path_collision_free(self, p1, p2):
        dist = self.distance(p1, p2)
        steps = int(dist / self.resolution) + 1

        for i in range(steps + 1):
            t = i / float(max(steps, 1))
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self.is_collision_free(x, y):
                return False
        return True

    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def sample_random(self):
        if random.random() < self.goal_sample_rate:
            return self.goal

        for _ in range(100):
            x = random.uniform(self.x_min + self.robot_radius,
                             self.x_max - self.robot_radius)
            y = random.uniform(self.y_min + self.robot_radius,
                             self.y_max - self.robot_radius)
            if self.is_collision_free(x, y):
                return (x, y)
        return self.goal

    def nearest_node(self, point):
        distances = [self.distance(node, point) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_point, to_point):
        dist = self.distance(from_point, to_point)
        if dist <= self.step_size:
            return to_point

        ratio = self.step_size / dist
        x = from_point[0] + ratio * (to_point[0] - from_point[0])
        y = from_point[1] + ratio * (to_point[1] - from_point[1])
        return (x, y)

    def near_nodes(self, point):
        n = len(self.nodes)
        r = min(self.search_radius * np.sqrt(np.log(n) / n), self.step_size)
        near_indices = []

        for i, node in enumerate(self.nodes):
            if self.distance(node, point) <= r:
                near_indices.append(i)
        return near_indices

    def plan(self):
        """Plan path and store tree structure"""
        goal_node_idx = None

        for iteration in range(self.max_iter):
            # Sample random point
            rand_point = self.sample_random()

            # Find nearest node
            nearest_idx = self.nearest_node(rand_point)
            nearest_point = self.nodes[nearest_idx]

            # Steer toward random point
            new_point = self.steer(nearest_point, rand_point)

            # Check collision
            if not self.is_path_collision_free(nearest_point, new_point):
                continue

            # Add new node
            new_idx = len(self.nodes)
            self.nodes.append(new_point)

            # Find near nodes for rewiring
            near_indices = self.near_nodes(new_point)

            # Choose best parent
            min_cost = self.costs[nearest_idx] + self.distance(nearest_point, new_point)
            min_idx = nearest_idx

            for idx in near_indices:
                if idx in self.costs and self.is_path_collision_free(self.nodes[idx], new_point):
                    cost = self.costs[idx] + self.distance(self.nodes[idx], new_point)
                    if cost < min_cost:
                        min_cost = cost
                        min_idx = idx

            # Add edge for visualization
            self.edges.append((self.nodes[min_idx], new_point))
            self.parents[new_idx] = min_idx
            self.costs[new_idx] = min_cost

            # Rewire tree
            for idx in near_indices:
                if idx != min_idx and idx in self.costs:
                    cost = min_cost + self.distance(new_point, self.nodes[idx])
                    if cost < self.costs[idx]:
                        if self.is_path_collision_free(new_point, self.nodes[idx]):
                            # Update edge
                            old_parent = self.parents[idx]
                            self.parents[idx] = new_idx
                            self.costs[idx] = cost

            # Check if goal reached
            if self.distance(new_point, self.goal) <= self.goal_threshold:
                if goal_node_idx is None or min_cost < self.costs[goal_node_idx]:
                    goal_node_idx = new_idx

        # Extract path
        if goal_node_idx is not None:
            path = []
            current_idx = goal_node_idx
            while current_idx is not None:
                path.append(self.nodes[current_idx])
                current_idx = self.parents.get(current_idx)
            return list(reversed(path))

        return None


def create_rrt_tree_visualization():
    """Create beautiful RRT* tree exploration visualization"""

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Load configuration
    config_dir = '../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    # Use office01add environment
    env_name = 'office01add'
    dataset_path = '../mrpb_dataset'
    parser = MRPBMapParser(env_name, dataset_path)

    # Get test configuration
    env_tests = env_config['environments'][env_name]['tests']
    test_config = env_tests[0]
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color scheme
    colors = {
        'naive': '#3498DB',      # Bright blue
        'standard': '#E67E22',   # Orange
        'learnable': '#2ECC71',  # Green
        'obstacles': '#2C3E50',  # Dark gray
        'tree': '#95A5A6',       # Light gray
        'path': None,            # Will use method color
        'start': '#27AE60',      # Green
        'goal': '#C0392B',       # Red
        'explored': '#ECF0F1'    # Very light gray
    }

    # Method configurations
    methods = [
        {
            'name': 'Naive Planning',
            'radius': 0.17,
            'color': colors['naive'],
            'iterations': 1500
        },
        {
            'name': 'Standard CP',
            'radius': 0.17 + 0.32,  # Fixed margin
            'color': colors['standard'],
            'iterations': 2000
        },
        {
            'name': 'Learnable CP',
            'radius': 0.25,  # Average adaptive margin
            'color': colors['learnable'],
            'iterations': 1800
        }
    ]

    for idx, (ax, method) in enumerate(zip(axes, methods)):

        # Display occupancy grid as background
        occupancy_display = parser.occupancy_grid.copy()

        # Draw obstacles
        for y in range(occupancy_display.shape[0]):
            for x in range(occupancy_display.shape[1]):
                if occupancy_display[y, x] > 0:
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor=colors['obstacles'],
                                   edgecolor='none',
                                   alpha=0.9)
                    ax.add_patch(rect)

        # Create and run planner
        planner = RRTStarGridVisual(
            start=start,
            goal=goal,
            occupancy_grid=parser.occupancy_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=method['radius'],
            step_size=2.0,
            max_iter=method['iterations'],
            goal_threshold=1.0,
            search_radius=5.0
        )

        # Plan and get path
        path = planner.plan()

        # Draw RRT* tree edges
        for edge in planner.edges[::2]:  # Sample every other edge for clarity
            p1, p2 = edge
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                   color=colors['tree'],
                   linewidth=0.5,
                   alpha=0.3,
                   zorder=1)

        # Draw explored nodes
        nodes_array = np.array(planner.nodes)
        ax.scatter(nodes_array[::5, 0], nodes_array[::5, 1],  # Sample nodes
                  s=2,
                  color=colors['explored'],
                  alpha=0.6,
                  zorder=2)

        # Draw optimal path
        if path:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1],
                   color=method['color'],
                   linewidth=4,
                   alpha=0.9,
                   zorder=4,
                   solid_capstyle='round',
                   solid_joinstyle='round',
                   label='Optimal Path')

            # Add waypoint markers along path
            indices = np.linspace(0, len(path)-1, 8, dtype=int)
            for idx in indices[1:-1]:  # Skip start and goal
                ax.scatter(path[idx, 0], path[idx, 1],
                          s=40,
                          color=method['color'],
                          alpha=0.7,
                          zorder=5,
                          edgecolors='white',
                          linewidths=1)

        # Draw start and goal
        # Start
        ax.scatter(start[0], start[1],
                  s=250,
                  color=colors['start'],
                  marker='o',
                  edgecolors='white',
                  linewidths=3,
                  zorder=6,
                  label='Start')

        # Goal
        ax.scatter(goal[0], goal[1],
                  s=250,
                  color=colors['goal'],
                  marker='*',
                  edgecolors='white',
                  linewidths=3,
                  zorder=6,
                  label='Goal')

        # Set title
        ax.set_title(f'{method["name"]}\n({len(planner.nodes)} nodes explored)',
                    fontsize=13,
                    fontweight='bold',
                    pad=10)

        # Set axis properties
        ax.set_xlim([0, parser.occupancy_grid.shape[1]])
        ax.set_ylim([0, parser.occupancy_grid.shape[0]])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)

        # Style frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#34495E')
            spine.set_linewidth(2)

        # Add statistics box
        if path:
            path_length = sum(planner.distance(path[i], path[i+1])
                            for i in range(len(path)-1))
            stats_text = (f"Robot radius: {method['radius']:.2f}m\n"
                         f"Path length: {path_length:.1f}m\n"
                         f"Waypoints: {len(path)}")
            ax.text(0.02, 0.02, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.9))

    # Add legend to middle subplot
    legend_elements = [
        Line2D([0], [0], color=colors['tree'], linewidth=1, alpha=0.5,
               label='RRT* Tree'),
        Line2D([0], [0], color='#3498DB', linewidth=3,
               label='Optimal Path'),
        patches.Circle((0, 0), 0.1, facecolor=colors['explored'],
                      alpha=0.6, label='Explored Nodes'),
        patches.Circle((0, 0), 0.1, facecolor=colors['obstacles'],
                      label='Obstacles'),
    ]

    axes[1].legend(handles=legend_elements,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=4,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10)

    # Main title
    fig.suptitle('RRT* Tree Exploration: Impact of Safety Margins',
                fontsize=16,
                fontweight='bold',
                y=1.02)

    plt.tight_layout()

    # Save figures
    plt.savefig('rrt_tree_exploration.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig('rrt_tree_exploration.png', dpi=150, bbox_inches='tight')

    print("Saved RRT* tree exploration visualization")
    plt.show()


if __name__ == "__main__":
    create_rrt_tree_visualization()