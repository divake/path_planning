"""
Learnable Conformal Prediction: Adaptive safety margins based on learned features.
Uses a neural network to predict appropriate margins for different situations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, List, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import FeatureExtractor


class UncertaintyMLP(nn.Module):
    """Simple MLP for uncertainty prediction."""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        """
        Initialize MLP model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
        """
        super(UncertaintyMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class LearnableCPMethod:
    """Learnable CP with adaptive margins."""
    
    def __init__(self, confidence_level: float = 0.95,
                 min_margin: float = 0.1,
                 max_margin: float = 2.0,
                 model_path: Optional[str] = None):
        """
        Initialize learnable CP method.
        
        Args:
            confidence_level: Desired confidence level
            min_margin: Minimum allowed margin
            max_margin: Maximum allowed margin
            model_path: Path to pre-trained model
        """
        self.name = "learnable_cp"
        self.confidence_level = confidence_level
        self.min_margin = min_margin
        self.max_margin = max_margin
        
        # Initialize model
        self.model = UncertaintyMLP()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Feature extractor (will be set when applying)
        self.feature_extractor = None
        
        # Calibration threshold
        self.calibration_threshold = 0.5
    
    def apply(self, environment) -> Dict[str, Any]:
        """
        Apply learnable CP with adaptive margins.
        
        Args:
            environment: UncertaintyEnvironment instance
            
        Returns:
            Dictionary with method info
        """
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(environment)
        
        # Extract features for each obstacle
        adaptive_margins = []
        
        for obs in environment.original_obstacles:
            # Extract features around obstacle
            features = self.feature_extractor.extract_point_features(obs['center'])
            
            # Predict uncertainty score
            uncertainty_score = self.predict_uncertainty(features)
            
            # Convert score to margin
            margin = self.score_to_margin(uncertainty_score)
            adaptive_margins.append(margin)
        
        # Apply adaptive margins to obstacles
        environment.obstacles = []
        for i, obs in enumerate(environment.original_obstacles):
            inflated_obs = obs.copy()
            if i < len(adaptive_margins):
                inflated_obs['radius'] += adaptive_margins[i]
            environment.obstacles.append(inflated_obs)
        
        # Store for analysis
        self.last_margins = adaptive_margins
        
        return {
            'method': self.name,
            'margins': adaptive_margins,
            'avg_margin': np.mean(adaptive_margins) if adaptive_margins else 0,
            'confidence_level': self.confidence_level,
            'description': f'Adaptive margins from {self.min_margin} to {self.max_margin}'
        }
    
    def predict_uncertainty(self, features: np.ndarray) -> float:
        """
        Predict uncertainty score for given features.
        
        Args:
            features: Feature vector
            
        Returns:
            Uncertainty score between 0 and 1
        """
        self.model.eval()
        
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Predict
            score = self.model(x).item()
        
        return score
    
    def score_to_margin(self, score: float) -> float:
        """
        Convert uncertainty score to margin.
        
        Args:
            score: Uncertainty score (0-1)
            
        Returns:
            Margin in units
        """
        # Linear interpolation between min and max margin
        margin = self.min_margin + score * (self.max_margin - self.min_margin)
        
        # Apply calibration threshold if needed
        if score > self.calibration_threshold:
            margin *= 1.2  # Increase margin for high uncertainty
        
        return margin
    
    def train(self, training_data: List[Tuple[np.ndarray, float]],
             validation_data: Optional[List[Tuple[np.ndarray, float]]] = None,
             epochs: int = 100,
             learning_rate: float = 0.001):
        """
        Train the uncertainty model.
        
        Args:
            training_data: List of (features, label) pairs
            validation_data: Optional validation set
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.model.train()
        
        # Prepare data
        X_train = torch.FloatTensor([x for x, _ in training_data]).to(self.device)
        y_train = torch.FloatTensor([y for _, y in training_data]).unsqueeze(1).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if validation_data and epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    X_val = torch.FloatTensor([x for x, _ in validation_data]).to(self.device)
                    y_val = torch.FloatTensor([y for _, y in validation_data]).unsqueeze(1).to(self.device)
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    
                    print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                self.model.train()
    
    def calibrate(self, calibration_data: List[Tuple[np.ndarray, bool]]):
        """
        Calibrate the model using conformal prediction.
        
        Args:
            calibration_data: List of (features, collision) pairs
        """
        scores = []
        
        for features, had_collision in calibration_data:
            score = self.predict_uncertainty(features)
            scores.append((score, had_collision))
        
        # Sort by score
        scores.sort(key=lambda x: x[0])
        
        # Find threshold for desired confidence level
        target_coverage = self.confidence_level
        best_threshold = 0.5
        
        for threshold in np.linspace(0, 1, 100):
            covered = sum(1 for s, collision in scores 
                         if (s > threshold) or not collision)
            coverage = covered / len(scores)
            
            if coverage >= target_coverage:
                best_threshold = threshold
                break
        
        self.calibration_threshold = best_threshold
        print(f"Calibrated threshold: {best_threshold:.3f}")
    
    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'calibration_threshold': self.calibration_threshold,
            'min_margin': self.min_margin,
            'max_margin': self.max_margin,
            'confidence_level': self.confidence_level
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.calibration_threshold = checkpoint.get('calibration_threshold', 0.5)
            self.min_margin = checkpoint.get('min_margin', self.min_margin)
            self.max_margin = checkpoint.get('max_margin', self.max_margin)
            self.confidence_level = checkpoint.get('confidence_level', self.confidence_level)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get method parameters."""
        return {
            'name': self.name,
            'min_margin': self.min_margin,
            'max_margin': self.max_margin,
            'confidence_level': self.confidence_level,
            'calibration_threshold': self.calibration_threshold,
            'type': 'adaptive_margin'
        }