"""
Traditional Conformal Prediction: Fixed uniform safety margins.
Inflates all obstacles by a constant margin to ensure safety.
"""

import numpy as np
from typing import Dict, Any, Optional


class TraditionalCPMethod:
    """Traditional CP with fixed margins."""
    
    def __init__(self, fixed_margin: float = 2.0, confidence_level: float = 0.95):
        """
        Initialize traditional CP method.
        
        Args:
            fixed_margin: Uniform margin to add to all obstacles
            confidence_level: Desired confidence level (e.g., 0.95)
        """
        self.name = "traditional_cp"
        self.fixed_margin = fixed_margin
        self.confidence_level = confidence_level
        
    def apply(self, environment) -> Dict[str, Any]:
        """
        Apply traditional CP with fixed margins.
        
        Args:
            environment: UncertaintyEnvironment instance
            
        Returns:
            Dictionary with method info
        """
        # Apply uniform inflation to all obstacles
        environment.apply_uncertainty(
            method='traditional_cp',
            fixed_margin=self.fixed_margin
        )
        
        return {
            'method': self.name,
            'margin': self.fixed_margin,
            'confidence_level': self.confidence_level,
            'description': f'Fixed margin of {self.fixed_margin} units'
        }
    
    def calibrate(self, calibration_data: Optional[np.ndarray] = None):
        """
        Calibrate the fixed margin based on calibration data.
        
        Args:
            calibration_data: Array of collision distances from calibration set
        """
        if calibration_data is not None and len(calibration_data) > 0:
            # Use quantile of calibration data as margin
            quantile = self.confidence_level
            self.fixed_margin = np.quantile(calibration_data, quantile)
            print(f"Calibrated margin to {self.fixed_margin:.2f} for {self.confidence_level} confidence")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters."""
        return {
            'name': self.name,
            'margin': self.fixed_margin,
            'confidence_level': self.confidence_level,
            'type': 'fixed_margin'
        }