#!/usr/bin/env python3
"""
RRT* for Learnable CP with adaptive safety margins based on obstacle proximity
"""

import os
import sys
import math
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection

sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
from mrpb_map_parser import MRPBMapParser

# Import from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretty_plot import set_pub_style, make_fig, draw_map, draw_tree, draw_start_goal, add_axes_labels, save_fig


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtStarLearnableCP:
    def __init__(self, x_start, x_goal, step_len, goal_sample_rate,
                 search_radius, iter_max, occupancy_grid, base_radius, resolution):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.occupancy_grid = occupancy_grid
        self.base_radius = base_radius  # 0.17m base radius
        self.resolution = resolution

        # For adaptive radius, use max possible radius for collision checking
        self.max_radius = base_radius + 0.35  # 0.17 + 0.35 = 0.52m max
        self.robot_radius_pixels = int(self.max_radius / resolution)

        self.x_range = (0, occupancy_grid.shape[1])
        self.y_range = (0, occupancy_grid.shape[0])

        self.goal_reached = False
        self.goal_node = None

        # Pre-compute collision map and distance field
        print("Pre-computing collision map and distance field...")
        self.collision_map = self.compute_collision_map()
        self.distance_field = self.compute_distance_field()
        print(f"Distance field computed: min={np.min(self.distance_field):.1f}, max={np.max(self.distance_field):.1f} pixels")

    def compute_collision_map(self):
        """Pre-compute dilated obstacle map for fast collision checking"""
        from scipy.ndimage import binary_dilation

        obstacles = (self.occupancy_grid > 50).astype(np.uint8)
        y, x = np.ogrid[-self.robot_radius_pixels:self.robot_radius_pixels+1,
                        -self.robot_radius_pixels:self.robot_radius_pixels+1]
        kernel = x**2 + y**2 <= self.robot_radius_pixels**2
        dilated = binary_dilation(obstacles, structure=kernel)

        return dilated

    def compute_distance_field(self):
        """Compute distance from each point to nearest obstacle"""
        from scipy.ndimage import distance_transform_edt

        # Create binary obstacle map
        obstacles = (self.occupancy_grid > 50).astype(np.uint8)

        # Compute Euclidean distance transform
        distance_field = distance_transform_edt(1 - obstacles)

        return distance_field

    def get_adaptive_tau(self, x, y):
        """
        Get adaptive tau based on distance to nearest obstacle
        Returns tau in meters
        """
        x, y = int(x), int(y)

        # Boundary check
        if x < 0 or x >= self.distance_field.shape[1] or y < 0 or y >= self.distance_field.shape[0]:
            return 0.35  # Max tau for out of bounds

        # Get distance to nearest obstacle in pixels
        dist_pixels = self.distance_field[y, x]

        # Convert to meters (subtract robot base radius)
        dist_meters = (dist_pixels * self.resolution) - self.base_radius

        # Three-level adaptive tau based on clearance
        if dist_meters <= 0.5:  # Very close to obstacle (within 50cm clearance)
            tau = 0.30  # Maximum safety margin (reduced from 0.35)
        elif dist_meters <= 1.5:  # Medium distance (50cm to 1.5m clearance)
            tau = 0.20  # Medium safety margin
        else:  # Far from obstacles (>1.5m clearance)
            tau = 0.10  # Minimum safety margin

        return tau

    def get_adaptive_radius(self, x, y):
        """Get adaptive robot radius based on position"""
        tau = self.get_adaptive_tau(x, y)
        return self.base_radius + tau

    def is_collision_adaptive(self, node1, node2):
        """Collision check with adaptive radius along the path"""
        x1, y1 = int(node1.x), int(node1.y)
        x2, y2 = int(node2.x), int(node2.y)

        # Check points along the line with adaptive radius
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = max(int(dist / 2), 1)

        for i in range(steps + 1):
            t = i / float(steps)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Get adaptive radius at this point
            adaptive_radius = self.get_adaptive_radius(x, y)
            radius_pixels = int(adaptive_radius / self.resolution)

            # Check collision with adaptive radius
            if not self.is_inside_adaptive(x, y, radius_pixels):
                return True

        return False

    def is_inside_adaptive(self, x, y, radius_pixels):
        """Check if point is collision-free with given radius"""
        x, y = int(x), int(y)

        if (x - radius_pixels < 0 or x + radius_pixels >= self.occupancy_grid.shape[1] or
            y - radius_pixels < 0 or y + radius_pixels >= self.occupancy_grid.shape[0]):
            return False

        for dx in range(-radius_pixels, radius_pixels + 1):
            for dy in range(-radius_pixels, radius_pixels + 1):
                if dx*dx + dy*dy <= radius_pixels * radius_pixels:
                    check_x = x + dx
                    check_y = y + dy
                    if self.occupancy_grid[check_y, check_x] > 50:
                        return False

        return True

    def planning(self):
        """RRT* planning with adaptive radius"""
        start_time = time.time()

        for k in range(self.iter_max):
            if k % 2500 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {k}/{self.iter_max}, tree size: {len(self.vertex)}, "
                      f"elapsed: {elapsed:.1f}s, goal reached: {self.goal_reached}")

            # Early termination if goal reached and path is good
            if self.goal_reached and k > 30000:
                if k % 500 == 0:
                    path_cost = self.cost(self.goal_node)
                    if path_cost < self.step_len * 150:
                        print(f"Good path found at iteration {k}, cost: {path_cost:.1f}")
                        break

            # Generate random node
            node_rand = self.generate_random_node()
            node_near = self.nearest_neighbor_fast(node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision_adaptive(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

                # Check goal connection
                if not self.goal_reached:
                    dist_to_goal = math.hypot(node_new.x - self.s_goal.x,
                                            node_new.y - self.s_goal.y)
                    if dist_to_goal <= self.step_len * 1.5:
                        if not self.is_collision_adaptive(node_new, self.s_goal):
                            self.goal_reached = True
                            self.goal_node = node_new
                            print(f"GOAL REACHED at iteration {k}!")

        # Extract path
        if self.goal_reached and self.goal_node:
            self.path = self.extract_path(self.goal_node, True)
        else:
            print("Goal not reached, finding closest approach...")
            closest_node = self.find_closest_to_goal()
            self.path = self.extract_path(closest_node, False)

        elapsed = time.time() - start_time
        print(f"Planning completed in {elapsed:.1f}s")
        return self.path

    def generate_random_node(self):
        """Biased random sampling"""
        if np.random.random() > self.goal_sample_rate:
            # Sample from free space
            for _ in range(10):
                x = np.random.uniform(self.x_range[0] + self.robot_radius_pixels,
                                    self.x_range[1] - self.robot_radius_pixels)
                y = np.random.uniform(self.y_range[0] + self.robot_radius_pixels,
                                    self.y_range[1] - self.robot_radius_pixels)

                # Check with adaptive radius at this point
                adaptive_radius = self.get_adaptive_radius(x, y)
                radius_pixels = int(adaptive_radius / self.resolution)

                if self.is_inside_adaptive(x, y, radius_pixels):
                    return Node((x, y))

            # Fallback
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            return Node((x, y))
        return self.s_goal

    def nearest_neighbor_fast(self, node):
        """Fast nearest neighbor"""
        return min(self.vertex, key=lambda n: math.hypot(n.x - node.x, n.y - node.y))

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                        node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        return node_new

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len * 2)

        neighbors = []
        for i, nd in enumerate(self.vertex):
            dist = math.hypot(nd.x - node_new.x, nd.y - node_new.y)
            if dist <= r and not self.is_collision_adaptive(node_new, nd):
                neighbors.append(i)

        return neighbors

    def choose_parent(self, node_new, neighbor_index):
        if not neighbor_index:
            return

        costs = []
        for i in neighbor_index:
            costs.append(self.get_new_cost(self.vertex[i], node_new))

        min_idx = neighbor_index[np.argmin(costs)]
        node_new.parent = self.vertex[min_idx]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def find_closest_to_goal(self):
        """Find node closest to goal that can connect"""
        best_node = None
        best_cost = float('inf')

        for node in self.vertex:
            dist = math.hypot(node.x - self.s_goal.x, node.y - self.s_goal.y)
            if dist < best_cost:
                if not self.is_collision_adaptive(node, self.s_goal):
                    best_cost = dist
                    best_node = node

        if best_node is None:
            best_node = min(self.vertex, key=lambda n: math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y))

        return best_node

    def extract_path(self, node_end, include_goal):
        """Extract path from node to start"""
        path = []

        if include_goal:
            path.append([self.s_goal.x, self.s_goal.y])

        node = node_end
        while node:
            path.append([node.x, node.y])
            node = node.parent

        return path

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)
        return self.cost(node_start) + dist

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent
        return cost

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def draw_adaptive_safety_corridor(ax, path, rrt_planner):
    """Draw adaptive safety corridor with varying width based on obstacle proximity"""
    if not path or len(path) < 2:
        return

    # Sample points along the path for smoother corridor
    sampled_path = []
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(int(dist / 5), 2)  # Sample every 5 pixels

        for j in range(steps):
            t = j / float(steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            sampled_path.append([x, y])
    sampled_path.append(path[-1])

    def get_perpendicular(p1, p2, dist):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return (0, 0)
        perp_x = -dy / length * dist
        perp_y = dx / length * dist
        return (perp_x, perp_y)

    # Build adaptive boundaries
    left_boundaries = {0.1: [], 0.2: [], 0.35: []}
    right_boundaries = {0.1: [], 0.2: [], 0.35: []}
    colors = {0.1: (0.6, 0.8, 0.6, 0.25),   # Light green for safe areas
              0.2: (0.8, 0.8, 0.4, 0.25),   # Yellow for medium
              0.3: (0.8, 0.6, 0.6, 0.25)}    # Light red for dangerous (updated to 0.3)

    # Create segments based on tau values
    segments = []
    current_segment = []
    current_tau = None

    for i, point in enumerate(sampled_path):
        tau_meters = rrt_planner.get_adaptive_tau(point[0], point[1])

        if current_tau != tau_meters:
            if current_segment:
                segments.append((current_tau, current_segment))
            current_segment = [point]
            current_tau = tau_meters
        else:
            current_segment.append(point)

    if current_segment:
        segments.append((current_tau, current_segment))

    # Draw each segment with its corresponding tau
    for tau, segment in segments:
        if len(segment) < 2:
            continue

        tau_pixels = int(tau / rrt_planner.resolution)
        left_boundary = []
        right_boundary = []

        for i in range(len(segment)):
            if i == 0 and len(segment) > 1:
                perp = get_perpendicular(segment[0], segment[1], tau_pixels)
            elif i == len(segment) - 1:
                perp = get_perpendicular(segment[-2], segment[-1], tau_pixels)
            else:
                perp1 = get_perpendicular(segment[i-1], segment[i], tau_pixels)
                perp2 = get_perpendicular(segment[i], segment[i+1], tau_pixels)
                perp = ((perp1[0] + perp2[0])/2, (perp1[1] + perp2[1])/2)

            left_boundary.append((segment[i][0] + perp[0], segment[i][1] + perp[1]))
            right_boundary.append((segment[i][0] - perp[0], segment[i][1] - perp[1]))

        # Draw this segment's corridor
        corridor_vertices = left_boundary + right_boundary[::-1]

        # Use color based on tau value
        color = colors.get(tau, (0.7, 0.7, 0.7, 0.2))

        corridor_poly = Polygon(corridor_vertices,
                               facecolor=color,
                               edgecolor='none',
                               zorder=3)
        ax.add_patch(corridor_poly)

        # Draw boundaries
        left_array = np.array(left_boundary)
        right_array = np.array(right_boundary)

        # Boundary colors based on tau
        if tau >= 0.25:  # Dangerous - red boundaries (updated threshold)
            boundary_color = (0.8, 0.2, 0.2, 0.7)
        elif tau >= 0.15:  # Medium - orange boundaries
            boundary_color = (0.8, 0.5, 0.2, 0.7)
        else:  # Safe - green boundaries
            boundary_color = (0.2, 0.6, 0.2, 0.7)

        # Upper boundary
        ax.plot(left_array[:, 0], left_array[:, 1],
               color=boundary_color,
               linewidth=0.5,
               linestyle='-',
               zorder=3)

        # Lower boundary
        ax.plot(right_array[:, 0], right_array[:, 1],
               color=boundary_color,
               linewidth=0.5,
               linestyle='-',
               zorder=3)


def draw_path_with_outline(ax, path, lw=0.8):
    """Draw the main path with white outline"""
    if not path:
        return
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    # White outline
    ax.plot(xs, ys, "-", linewidth=lw+0.6, color="white", alpha=0.95, zorder=4)
    # Red path
    ax.plot(xs, ys, "-", linewidth=lw, color=(0.82, 0.10, 0.10), zorder=5)


def compute_path_length(path, resolution):
    """Compute path length in meters"""
    if len(path) < 2:
        return 0
    length = 0
    for i in range(len(path) - 1):
        length += math.hypot(path[i+1][0] - path[i][0],
                           path[i+1][1] - path[i][1])
    return length * resolution


def main():
    """Run Learnable CP RRT* with adaptive safety margins"""

    # Load MRPB map
    print("Loading MRPB map...")
    config_dir = '../../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    env_name = 'shopping_mall'
    dataset_path = '../../mrpb_dataset'
    parser = MRPBMapParser(env_name, dataset_path)

    # Get test points
    env_tests = env_config['environments'][env_name]['tests']
    test_config = env_tests[0]

    def world_to_pixel(x, y, origin, resolution):
        px = int((x - origin[0]) / resolution)
        py = int((y - origin[1]) / resolution)
        return (px, py)

    start_world = test_config['start']
    goal_world = test_config['goal']

    x_start = world_to_pixel(start_world[0], start_world[1], parser.origin, parser.resolution)
    x_goal = world_to_pixel(goal_world[0], goal_world[1], parser.origin, parser.resolution)

    print(f"Start: {x_start} (pixels)")
    print(f"Goal: {x_goal} (pixels)")

    # Learnable CP parameters
    BASE_RADIUS = 0.17  # Base robot radius in meters

    # RRT* parameters
    step_len = 12
    goal_sample_rate = 0.20
    search_radius = 35
    iter_max = 50000

    # Run RRT*
    print(f"\nRunning Learnable CP RRT* with adaptive safety margins...")
    print(f"  Base radius: {BASE_RADIUS:.2f}m")
    print(f"  Adaptive tau: 0.10m (safe), 0.20m (medium), 0.30m (dangerous)")

    rrt_star = RrtStarLearnableCP(x_start, x_goal, step_len, goal_sample_rate,
                                  search_radius, iter_max, parser.occupancy_grid,
                                  BASE_RADIUS, parser.resolution)

    path = rrt_star.planning()

    # Statistics
    nodes_explored = len(rrt_star.vertex)
    waypoints = len(path) if path else 0
    path_length_m = compute_path_length(path, parser.resolution) if path else 0

    # Calculate tau statistics along path
    if path:
        tau_stats = {0.1: 0, 0.2: 0, 0.3: 0}
        for point in path:
            tau = rrt_star.get_adaptive_tau(point[0], point[1])
            if tau <= 0.15:
                tau_stats[0.1] += 1
            elif tau <= 0.25:
                tau_stats[0.2] += 1
            else:
                tau_stats[0.3] += 1

        print(f"\nAdaptive tau distribution along path:")
        print(f"  Safe (τ=0.10m): {tau_stats[0.1]} waypoints ({100*tau_stats[0.1]/waypoints:.1f}%)")
        print(f"  Medium (τ=0.20m): {tau_stats[0.2]} waypoints ({100*tau_stats[0.2]/waypoints:.1f}%)")
        print(f"  Dangerous (τ=0.30m): {tau_stats[0.3]} waypoints ({100*tau_stats[0.3]/waypoints:.1f}%)")

    print(f"\nResults:")
    print(f"  Nodes explored: {nodes_explored}")
    print(f"  Path waypoints: {waypoints}")
    print(f"  Path length: {path_length_m:.2f} meters")
    print(f"  Status: {'COMPLETE PATH' if rrt_star.goal_reached else 'PARTIAL PATH'}")

    # Create visualization
    fig, ax = make_fig(size="single")

    # Draw map
    draw_map(ax, parser.occupancy_grid, wall_px=2)

    # Draw RRT tree
    draw_tree(ax, rrt_star.vertex, lw=0.6, every=1, color=(0.16, 0.60, 0.16, 0.45))

    # Draw adaptive safety corridor
    if path:
        draw_adaptive_safety_corridor(ax, path, rrt_star)

    # Draw path
    draw_path_with_outline(ax, path, lw=0.8)

    # Draw start and goal
    draw_start_goal(ax, x_start, x_goal, s=6.0)

    # Add axis labels
    add_axes_labels(ax, parser.occupancy_grid.shape, parser.resolution, parser.origin)

    # Add legend for adaptive tau
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.6, 0.8, 0.6, 0.5), label='τ=0.10m (Safe)'),
        Patch(facecolor=(0.8, 0.8, 0.4, 0.5), label='τ=0.20m (Medium)'),
        Patch(facecolor=(0.8, 0.6, 0.6, 0.5), label='τ=0.30m (Dangerous)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.8)

    # Save figure
    output_file = "learnable_cp/rrt_star_learnable_cp"
    save_fig(fig, output_file)
    plt.close(fig)

    print(f"\nVisualization saved to {output_file}")


if __name__ == '__main__':
    main()