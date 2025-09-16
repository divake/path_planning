#!/usr/bin/env python3
"""
RRT* for Naive method - no safety margins, just base robot radius
Uses same optimized structure as Standard CP and Learnable CP
"""

import os
import sys
import math
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt

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


class RrtStarNaive:
    def __init__(self, x_start, x_goal, step_len, goal_sample_rate,
                 search_radius, iter_max, occupancy_grid, robot_radius, resolution):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.occupancy_grid = occupancy_grid
        self.robot_radius_pixels = int(robot_radius / resolution)
        self.x_range = (0, occupancy_grid.shape[1])
        self.y_range = (0, occupancy_grid.shape[0])

        self.goal_reached = False
        self.goal_node = None

        # Pre-compute collision map with dilation for robot radius
        print("Pre-computing collision map...")
        self.collision_map = self.compute_collision_map()

    def compute_collision_map(self):
        """Pre-compute dilated obstacle map for fast collision checking"""
        # Create binary obstacle map
        obstacles = (self.occupancy_grid > 50).astype(np.uint8)

        # Dilate obstacles by robot radius using circular kernel
        from scipy.ndimage import binary_dilation

        # Create circular structuring element
        y, x = np.ogrid[-self.robot_radius_pixels:self.robot_radius_pixels+1,
                        -self.robot_radius_pixels:self.robot_radius_pixels+1]
        kernel = x**2 + y**2 <= self.robot_radius_pixels**2

        # Dilate obstacles
        dilated = binary_dilation(obstacles, structure=kernel)

        return dilated

    def is_collision(self, node1, node2):
        """Fast collision check using pre-computed map"""
        x1, y1 = int(node1.x), int(node1.y)
        x2, y2 = int(node2.x), int(node2.y)

        # Bresenham's line algorithm for fast line checking
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            # Check bounds and collision
            if (x < 0 or x >= self.collision_map.shape[1] or
                y < 0 or y >= self.collision_map.shape[0] or
                self.collision_map[y, x]):
                return True

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return False

    def is_inside(self, x, y):
        """Fast point collision check"""
        x, y = int(x), int(y)
        if (x < 0 or x >= self.collision_map.shape[1] or
            y < 0 or y >= self.collision_map.shape[0]):
            return False
        return not self.collision_map[y, x]

    def planning(self):
        """Optimized RRT* planning"""
        start_time = time.time()

        # Pre-allocate arrays for batch operations
        vertex_array = np.array([[self.s_start.x, self.s_start.y]])

        for k in range(self.iter_max):
            # Progress update
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

            # Find nearest using vectorized operations
            if len(self.vertex) > 100 and k % 10 == 0:
                # Batch nearest neighbor for efficiency
                vertex_array = np.array([[n.x, n.y] for n in self.vertex])

            node_near = self.nearest_neighbor_fast(node_rand, vertex_array if len(self.vertex) > 100 else None)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.is_collision(node_near, node_new):
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
                        if not self.is_collision(node_new, self.s_goal):
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
                if self.is_inside(x, y):
                    return Node((x, y))
            # Fallback to any random point
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            return Node((x, y))
        return self.s_goal

    def nearest_neighbor_fast(self, node, vertex_array=None):
        """Fast nearest neighbor"""
        if vertex_array is not None and len(vertex_array) > 0:
            # Vectorized distance computation
            dists = np.sum((vertex_array - np.array([node.x, node.y]))**2, axis=1)
            idx = np.argmin(dists)
            return self.vertex[idx]
        else:
            # Fallback to simple method
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
            if dist <= r and not self.is_collision(node_new, nd):
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
                if not self.is_collision(node, self.s_goal):
                    best_cost = dist
                    best_node = node

        if best_node is None:
            # Just return closest node
            best_node = min(self.vertex, key=lambda n: math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y))

        return best_node

    def extract_path(self, node_end, include_goal):
        """Extract path from node to start"""
        path = []

        # Add goal if we reached it
        if include_goal:
            path.append([self.s_goal.x, self.s_goal.y])

        # Build path from node to start
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
    """Run Naive RRT* with no safety margins"""

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

    # Naive parameters - only base robot radius, no safety margin
    NAIVE_RADIUS = 0.17  # Base robot radius in meters (no tau)

    # RRT* parameters - same as other methods for fair comparison
    step_len = 12  # Same as Standard CP and Learnable CP
    goal_sample_rate = 0.20  # Same as other methods
    search_radius = 35  # Same as other methods
    iter_max = 50000  # Same as other methods

    # Run RRT*
    print(f"\nRunning Naive RRT* (no safety margins)...")
    print(f"  Robot radius: {NAIVE_RADIUS:.2f}m (base only, no tau)")

    rrt_star = RrtStarNaive(x_start, x_goal, step_len, goal_sample_rate,
                            search_radius, iter_max, parser.occupancy_grid,
                            NAIVE_RADIUS, parser.resolution)

    path = rrt_star.planning()

    # Statistics
    nodes_explored = len(rrt_star.vertex)
    waypoints = len(path) if path else 0
    path_length_m = compute_path_length(path, parser.resolution) if path else 0

    print(f"\nResults:")
    print(f"  Nodes explored: {nodes_explored}")
    print(f"  Path waypoints: {waypoints}")
    print(f"  Path length: {path_length_m:.2f} meters")
    print(f"  Status: {'COMPLETE PATH' if rrt_star.goal_reached else 'PARTIAL PATH'}")

    # Create visualization - same style as Standard CP and Learnable CP
    fig, ax = make_fig(size="single")

    # Draw map
    draw_map(ax, parser.occupancy_grid, wall_px=2)

    # Draw RRT tree - same style as other methods
    draw_tree(ax, rrt_star.vertex, lw=0.6, every=1, color=(0.16, 0.60, 0.16, 0.45))

    # No safety corridor for Naive method - this is the key difference!

    # Draw path - same style as other methods
    draw_path_with_outline(ax, path, lw=0.8)

    # Draw start and goal
    draw_start_goal(ax, x_start, x_goal, s=6.0)

    # Add axis labels
    add_axes_labels(ax, parser.occupancy_grid.shape, parser.resolution, parser.origin)

    # Save figure
    output_dir = "naive"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/rrt_star_naive"
    save_fig(fig, output_file)
    plt.close(fig)

    print(f"\nVisualization saved to {output_file}")


if __name__ == '__main__':
    main()