#!/usr/bin/env python3
"""
RRT* for MRPB maps - Publication-ready visualization
Uses pretty_plot module for clean IEEE/ICRA style figures
"""

import os
import sys
import math
import numpy as np
import yaml

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
        steps = int(dist / 2) + 1

        for i in range(steps + 1):
            t = i / float(max(steps, 1))
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


class RrtStarMRPB:
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

    def planning(self):
        """Main RRT* planning loop"""
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 1000 == 0:
                print(f"Iteration {k}/{self.iter_max}")

            if node_new and not self.utils.is_collision(node_near, node_new):
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])

        return self.path

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

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                        if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            if len(cost_list) > 0:
                return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)
        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta
        if np.random.random() > goal_sample_rate:
            # Uniform random sampling across entire space
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

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

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
    """Run RRT* on MRPB map with publication-ready visualization"""

    # Load MRPB map
    print("Loading MRPB map...")
    config_dir = '../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    env_name = 'office01add'
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

    # Create environment and utils
    env = MRPBEnv(parser.occupancy_grid, parser.resolution)
    utils = UtilsMRPB(parser.occupancy_grid, robot_radius=0.17, resolution=parser.resolution)

    # RRT* parameters
    step_len = 10  # pixels
    goal_sample_rate = 0.05  # Lower to explore more (was 0.10)
    search_radius = 25  # pixels (increased for better rewiring)
    iter_max = 5000  # More iterations for better coverage (was 3000)

    # Run RRT*
    print("\nRunning RRT* (Naive method with r=0.17m)...")
    rrt_star = RrtStarMRPB(x_start, x_goal, step_len, goal_sample_rate,
                           search_radius, iter_max, env, utils)

    path = rrt_star.planning()

    # Calculate statistics
    nodes_explored = len(rrt_star.vertex)
    waypoints = len(path)
    path_length_m = compute_path_length(path, parser.resolution)

    print(f"\nPath found with {nodes_explored} nodes explored")
    print(f"Path has {waypoints} waypoints")
    print(f"Path length: {path_length_m:.2f} meters")

    # Render publication-ready figure with tree
    render_rrt_figure(
        occupancy_grid=parser.occupancy_grid,
        start=x_start,
        goal=x_goal,
        nodes=rrt_star.vertex,
        path=path,
        size="single",          # Single column figure
        wall_px=2,              # Thick walls
        show_tree=True,         # Show RRT tree
        out_base=f"rrt_star_{env_name}_clean",
        resolution=parser.resolution,
        origin=parser.origin
    )


if __name__ == '__main__':
    main()