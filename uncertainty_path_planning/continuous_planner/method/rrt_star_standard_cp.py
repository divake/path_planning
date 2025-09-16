#!/usr/bin/env python3
"""
Optimized RRT* for Standard CP with proper path extraction and parallel sampling
"""

import os
import sys
import math
import numpy as np
import yaml
from multiprocessing import Pool, cpu_count
import time

sys.path.append('..')
sys.path.append('../..')
from mrpb_map_parser import MRPBMapParser
from pretty_plot import render_rrt_figure


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class MRPBEnv:
    """Environment from MRPB map"""
    def __init__(self, occupancy_grid, resolution):
        self.occupancy_grid = occupancy_grid
        self.resolution = resolution
        self.x_range = (0, occupancy_grid.shape[1])
        self.y_range = (0, occupancy_grid.shape[0])


class UtilsMRPB:
    """Utils for collision checking on MRPB maps"""
    def __init__(self, occupancy_grid, robot_radius=0.17, resolution=0.05):
        self.occupancy_grid = occupancy_grid
        self.robot_radius = robot_radius
        self.resolution = resolution
        self.robot_radius_pixels = int(robot_radius / resolution)
        self.delta = self.robot_radius_pixels

    def is_collision(self, node1, node2):
        """Check collision between two nodes"""
        dist = math.hypot(node2.x - node1.x, node2.y - node1.y)
        steps = max(int(dist / 2), 1)  # Check every 2 pixels

        for i in range(steps + 1):
            t = i / float(steps)
            x = int(node1.x + t * (node2.x - node1.x))
            y = int(node1.y + t * (node2.y - node1.y))

            if not self.is_inside(x, y):
                return True

        return False

    def is_inside(self, x, y):
        """Check if point is collision-free"""
        if (x - self.robot_radius_pixels < 0 or
            x + self.robot_radius_pixels >= self.occupancy_grid.shape[1] or
            y - self.robot_radius_pixels < 0 or
            y + self.robot_radius_pixels >= self.occupancy_grid.shape[0]):
            return False

        for dx in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
            for dy in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                if dx*dx + dy*dy <= self.robot_radius_pixels * self.robot_radius_pixels:
                    check_x = x + dx
                    check_y = y + dy
                    if self.occupancy_grid[check_y, check_x] > 50:
                        return False

        return True


class RrtStarOptimized:
    def __init__(self, x_start, x_goal, step_len, goal_sample_rate,
                 search_radius, iter_max, env, utils):
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env
        self.utils = utils
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range

        # Track if we've reached the goal
        self.goal_reached = False
        self.goal_node = None

    def planning(self):
        """Main RRT* planning loop with proper termination"""
        print(f"Starting RRT* with max {self.iter_max} iterations...")
        start_time = time.time()

        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 5000 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {k}/{self.iter_max}, tree size: {len(self.vertex)}, "
                      f"elapsed: {elapsed:.1f}s, goal reached: {self.goal_reached}")

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

                # Check if we reached the goal
                if not self.goal_reached:
                    dist_to_goal = math.hypot(node_new.x - self.s_goal.x,
                                            node_new.y - self.s_goal.y)
                    if dist_to_goal <= self.step_len:
                        # Check if we can connect to goal
                        if not self.utils.is_collision(node_new, self.s_goal):
                            self.goal_reached = True
                            self.goal_node = node_new
                            print(f"GOAL REACHED at iteration {k}!")
                            # Continue for a bit to optimize the path
                            if k < self.iter_max - 1000:
                                self.iter_max = min(k + 1000, self.iter_max)

        # Extract path
        if self.goal_reached and self.goal_node:
            # Build path from goal node
            self.path = self.extract_path_from_node(self.goal_node)
            # Add the actual goal
            self.path.append([self.s_goal.x, self.s_goal.y])
        else:
            # Find closest node to goal and build partial path
            print("WARNING: Goal not reached, finding closest approach...")
            closest_idx = self.find_closest_to_goal()
            if closest_idx >= 0:
                self.path = self.extract_path_from_node(self.vertex[closest_idx])

        elapsed = time.time() - start_time
        print(f"Planning completed in {elapsed:.1f}s")
        return self.path

    def extract_path_from_node(self, node_end):
        """Extract path from a specific node to start"""
        path = []
        node = node_end
        while node is not None:
            path.append([node.x, node.y])
            node = node.parent
        return list(reversed(path))

    def find_closest_to_goal(self):
        """Find the node closest to goal that has a clear path"""
        min_dist = float('inf')
        closest_idx = -1

        for i, node in enumerate(self.vertex):
            dist = math.hypot(node.x - self.s_goal.x, node.y - self.s_goal.y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                        node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start
        return node_new

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]
        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)
        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta
        if np.random.random() > goal_sample_rate:
            x = np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta)
            y = np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)
            return Node((x, y))
        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)
        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                           not self.utils.is_collision(node_new, self.vertex[ind])]
        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                       for nd in node_list]))]

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
    """Run optimized Standard CP RRT*"""

    # Load MRPB map
    print("Loading MRPB map...")
    config_dir = '../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    env_name = 'shopping_mall'  # Use shopping_mall which works
    dataset_path = '../mrpb_dataset'
    parser = MRPBMapParser(env_name, dataset_path)

    # Get test points and convert to pixels
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

    # Standard CP parameters
    STANDARD_CP_RADIUS = 0.17 + 0.32  # Base + tau = 0.49m total

    # Create environment and utils
    env = MRPBEnv(parser.occupancy_grid, parser.resolution)
    utils = UtilsMRPB(parser.occupancy_grid, robot_radius=STANDARD_CP_RADIUS, resolution=parser.resolution)

    # RRT* parameters - more iterations for complete path
    step_len = 10  # pixels
    goal_sample_rate = 0.10  # 10% goal bias
    search_radius = 30  # pixels
    iter_max = 50000  # Much more iterations

    # Run RRT*
    print(f"\nRunning Optimized RRT* (Standard CP with r={STANDARD_CP_RADIUS:.2f}m)...")
    print(f"Using {cpu_count()} CPU cores available")

    rrt_star = RrtStarOptimized(x_start, x_goal, step_len, goal_sample_rate,
                                search_radius, iter_max, env, utils)

    path = rrt_star.planning()

    # Calculate statistics
    nodes_explored = len(rrt_star.vertex)
    waypoints = len(path) if path else 0
    path_length_m = compute_path_length(path, parser.resolution) if path else 0

    print(f"\nTree exploration: {nodes_explored} nodes explored")

    if waypoints > 2 and rrt_star.goal_reached:
        print(f"SUCCESS! Complete path found: {waypoints} waypoints, length: {path_length_m:.2f} meters")
    elif waypoints > 2:
        print(f"PARTIAL path found: {waypoints} waypoints, length: {path_length_m:.2f} meters")
        print("Path ends at closest reachable point to goal")
    else:
        print(f"FAILED: No valid path found")
        return

    # Render visualization
    render_rrt_figure(
        occupancy_grid=parser.occupancy_grid,
        start=x_start,
        goal=x_goal,
        nodes=rrt_star.vertex,
        path=path,
        size="single",
        wall_px=2,
        show_tree=True,
        out_base=f"standard_cp/rrt_star_standard_cp_optimized",
        resolution=parser.resolution,
        origin=parser.origin
    )

    print("\nVisualization saved to standard_cp/")


if __name__ == '__main__':
    main()