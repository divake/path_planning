"""
Naive method: No uncertainty consideration.
This serves as the baseline - uses original obstacles without any safety margin.
"""

import numpy as np
from typing import Dict, Any


class NaiveMethod:
    """Baseline method with no uncertainty handling."""
    
    def __init__(self):
        """Initialize naive method."""
        self.name = "naive"
        self.margin = 0.0
        
    def apply(self, environment) -> Dict[str, Any]:
        """
        Apply naive method (no changes to obstacles).
        
        Args:
            environment: UncertaintyEnvironment instance
            
        Returns:
            Dictionary with method info
        """
        # No inflation - use original obstacles as-is
        environment.apply_uncertainty(method='naive')
        
        return {
            'method': self.name,
            'margin': self.margin,
            'description': 'No uncertainty consideration (baseline)'
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters."""
        return {
            'name': self.name,
            'margin': self.margin,
            'type': 'baseline'
        }