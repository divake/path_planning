#!/usr/bin/env python3
"""
RRT* Planner Module
Clean implementation of RRT* algorithm for continuous path planning
"""

import numpy as np
import random
import math
from typing import List, Tuple, Optional, Dict


class Node:
    """Node for RRT* tree"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class RRTStar:
    """
    RRT* (Rapidly-exploring Random Tree Star) planner
    Continuous space path planning with asymptotic optimality
    """
    
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 obstacles: List,
                 bounds: Tuple[float, float, float, float] = (0, 50, 0, 30),
                 max_iter: int = 2000,
                 step_size: float = 2.0,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 5.0,
                 goal_threshold: float = 1.0):
        """
        Initialize RRT* planner
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            obstacles: List of obstacles (rectangles or functions)
            bounds: Environment bounds (x_min, x_max, y_min, y_max)
            max_iter: Maximum iterations
            step_size: Maximum step size for steering
            goal_sample_rate: Probability of sampling goal
            search_radius: Radius for rewiring
            goal_threshold: Distance threshold to consider goal reached
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.bounds = bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold
        
        self.nodes = [self.start]
        self.path = None
        
    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """
        Execute RRT* planning
        
        Returns:
            Path as list of (x, y) tuples, or None if no path found
        """
        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                sample = Node(self.goal.x, self.goal.y)
            else:
                sample = self._random_sample()
            
            # Find nearest node
            nearest = self._nearest_node(sample)
            
            # Steer towards sample
            new_node = self._steer(nearest, sample)
            
            # Check collision
            if not self._collision_check(nearest, new_node):
                # Find nearby nodes for rewiring
                near_nodes = self._near_nodes(new_node)
                
                # Choose best parent
                new_node = self._choose_parent(new_node, near_nodes)
                
                # Add to tree
                self.nodes.append(new_node)
                
                # Rewire tree
                self._rewire(new_node, near_nodes)
                
                # Check if reached goal
                if self._distance(new_node, self.goal) < self.goal_threshold:
                    self.path = self._extract_path(new_node)
                    return self.path
        
        # No path found
        return None
    
    def _random_sample(self) -> Node:
        """Generate random sample in bounds"""
        x = random.uniform(self.bounds[0], self.bounds[1])
        y = random.uniform(self.bounds[2], self.bounds[3])
        return Node(x, y)
    
    def _nearest_node(self, sample: Node) -> Node:
        """Find nearest node to sample"""
        distances = [self._distance(node, sample) for node in self.nodes]
        return self.nodes[np.argmin(distances)]
    
    def _steer(self, from_node: Node, to_node: Node) -> Node:
        """Steer from node towards another node"""
        dist = self._distance(from_node, to_node)
        
        if dist < self.step_size:
            new_node = Node(to_node.x, to_node.y)
        else:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = Node(
                from_node.x + self.step_size * math.cos(theta),
                from_node.y + self.step_size * math.sin(theta)
            )
        
        new_node.parent = from_node
        new_node.cost = from_node.cost + self._distance(from_node, new_node)
        return new_node
    
    def _near_nodes(self, node: Node) -> List[Node]:
        """Find nodes within search radius"""
        # Dynamic radius based on tree size (RRT* optimization)
        card = len(self.nodes)
        radius = min(self.search_radius * math.sqrt(math.log(card) / card), self.search_radius)
        
        near = []
        for n in self.nodes:
            if self._distance(n, node) < radius:
                near.append(n)
        return near
    
    def _choose_parent(self, new_node: Node, near_nodes: List[Node]) -> Node:
        """Choose best parent from nearby nodes"""
        if not near_nodes:
            return new_node
        
        # Find minimum cost parent
        min_cost = float('inf')
        best_parent = None
        
        for near_node in near_nodes:
            if not self._collision_check(near_node, new_node):
                cost = near_node.cost + self._distance(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near_node
        
        if best_parent:
            new_node.parent = best_parent
            new_node.cost = min_cost
        
        return new_node
    
    def _rewire(self, new_node: Node, near_nodes: List[Node]):
        """Rewire tree for optimality"""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
                
            cost = new_node.cost + self._distance(new_node, near_node)
            if cost < near_node.cost and not self._collision_check(new_node, near_node):
                near_node.parent = new_node
                near_node.cost = cost
    
    def _collision_check(self, from_node: Node, to_node: Node) -> bool:
        """
        Check if path from_node to to_node collides with obstacles
        
        Returns:
            True if collision, False if free
        """
        # Sample along the line
        steps = int(self._distance(from_node, to_node) / 0.1) + 1
        
        for i in range(steps + 1):
            t = i / float(steps)
            x = from_node.x + t * (to_node.x - from_node.x)
            y = from_node.y + t * (to_node.y - from_node.y)
            
            # Check against each obstacle
            for obs in self.obstacles:
                if isinstance(obs, tuple) and len(obs) == 4:
                    # Rectangle obstacle (x, y, width, height)
                    ox, oy, w, h = obs
                    if ox <= x <= ox + w and oy <= y <= oy + h:
                        return True
                elif callable(obs):
                    # Custom obstacle function
                    if obs(x, y):
                        return True
        
        return False
    
    def _distance(self, node1: Node, node2: Node) -> float:
        """Euclidean distance between nodes"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _extract_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """Extract path from start to goal"""
        path = []
        current = goal_node
        
        while current:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get start to goal
    
    def get_tree(self) -> List[Tuple[Node, Node]]:
        """Get tree edges for visualization"""
        edges = []
        for node in self.nodes:
            if node.parent:
                edges.append((node.parent, node))
        return edges
    
    def get_metrics(self) -> Dict:
        """Get planning metrics"""
        return {
            'nodes_explored': len(self.nodes),
            'path_found': self.path is not None,
            'path_length': self._path_length() if self.path else None,
            'path_cost': self.nodes[-1].cost if self.path else None
        }
    
    def _path_length(self) -> float:
        """Calculate path length"""
        if not self.path or len(self.path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(self.path) - 1):
            dx = self.path[i+1][0] - self.path[i][0]
            dy = self.path[i+1][1] - self.path[i][1]
            length += math.sqrt(dx*dx + dy*dy)
        
        return length