#!/usr/bin/env python3
"""
Cross-validation environments for testing generalization
These are NEW environments not used in training
"""

from typing import List, Tuple


def get_test_environments() -> dict:
    """
    Get test environments for cross-validation
    These are different from training environments
    """
    
    test_envs = {
        'zigzag': [
            # Zigzag corridor requiring maneuvering
            (10, 0, 5, 20),    # Bottom wall
            (25, 10, 5, 20),   # Top wall  
            (15, 5, 10, 3),    # Middle barrier 1
            (20, 22, 10, 3),   # Middle barrier 2
        ],
        
        'spiral': [
            # Spiral-like path
            (10, 10, 30, 3),   # Top horizontal
            (10, 10, 3, 10),   # Left vertical
            (10, 20, 20, 3),   # Bottom horizontal
            (30, 13, 3, 7),    # Right vertical
            (20, 13, 10, 3),   # Inner horizontal
        ],
        
        'random_forest': [
            # Random scattered obstacles (different from training)
            (8, 4, 3, 3),
            (18, 8, 4, 4), 
            (35, 6, 3, 3),
            (12, 18, 4, 3),
            (28, 20, 3, 4),
            (40, 15, 3, 3),
            (5, 25, 4, 2),
            (22, 14, 2, 4),
        ],
        
        'tight_gaps': [
            # Very narrow passages to test limits
            (15, 0, 2, 12),    # Bottom left wall
            (15, 18, 2, 12),   # Top left wall
            (33, 0, 2, 12),    # Bottom right wall
            (33, 18, 2, 12),   # Top right wall
            (24, 8, 2, 14),    # Middle wall
        ],
        
        'rooms': [
            # Room-like structure
            (0, 15, 20, 2),    # Left room wall
            (20, 0, 2, 30),    # Central divider
            (22, 15, 28, 2),   # Right room wall
            (20, 10, 8, 2),    # Door 1
            (20, 18, 8, 2),    # Door 2
        ]
    }
    
    return test_envs


def get_training_environments() -> dict:
    """
    Get the standard training environments
    """
    return {
        'passages': [
            (20, 0, 3, 15),    # Bottom wall
            (20, 20, 3, 10),   # Top wall
        ],
        
        'open': [
            (10, 8, 6, 6),     # Left obstacle
            (25, 5, 6, 6),     # Middle obstacle
            (35, 12, 6, 6),    # Right obstacle
            (20, 20, 6, 4),    # Top obstacle
        ],
        
        'narrow': [
            (0, 10, 20, 3),    # Left horizontal
            (0, 17, 20, 3),    # Left horizontal 2
            (30, 10, 20, 3),   # Right horizontal
            (30, 17, 20, 3),   # Right horizontal 2
            (20, 5, 3, 8),     # Vertical 1
            (20, 17, 3, 8),    # Vertical 2
            (35, 5, 3, 8),     # Vertical 3
            (35, 17, 3, 8),    # Vertical 4
        ],
    }