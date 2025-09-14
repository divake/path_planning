#!/usr/bin/env python3
"""
Training Pipeline for Learnable Conformal Prediction
Handles the training of the neural network for adaptive tau prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from datetime import datetime
import pickle
from tqdm import tqdm

from neural_network import AdaptiveTauNetwork, EnsembleTauNetwork
from feature_extractor import WaypointFeatureExtractor
from loss_function import AdaptiveTauLoss, QuantileLoss, WeightedMSELoss
from tau_calculator import AdaptiveTauCalculator


class PathPlanningDataset(Dataset):
    """
    Dataset for training adaptive tau predictor.
    Contains waypoints, features, and observed clearances from multiple environments.
    """
    
    def __init__(self, data_path: str, feature_extractor: WaypointFeatureExtractor):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to dataset file
            feature_extractor: Feature extraction module
        """
        self.feature_extractor = feature_extractor
        self.data = []
        
        # Load data
        self._load_data(data_path)
        
        logging.info(f"Loaded dataset with {len(self.data)} samples")
    
    def _load_data(self, data_path: str):
        """Load training data from file"""
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                raw_data = pickle.load(f)
        elif data_path.endswith('.npz'):
            raw_data = np.load(data_path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Process raw data into training samples
        for item in raw_data.get('samples', []):
            sample = {
                'features': torch.FloatTensor(item['features']),
                'clearance': torch.FloatTensor([item['clearance']]),
                'noise_type': item.get('noise_type', 'unknown'),
                'env_name': item.get('env_name', 'unknown'),
                'path_id': item.get('path_id', 0)
            }
            self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class LearnableCPTrainer:
    """
    Main trainer class for Learnable Conformal Prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._init_model()
        self._init_optimizer()
        self._init_loss()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'coverage': [],
            'avg_tau': []
        }
        
        # Paths
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logging.info(f"Trainer initialized on {self.device}")
    
    def _init_model(self):
        """Initialize neural network model"""
        model_config = self.config.get('model', {})
        
        if model_config.get('use_ensemble', False):
            self.model = EnsembleTauNetwork(
                num_models=model_config.get('num_models', 5),
                input_dim=model_config.get('input_dim', 20),
                hidden_dims=model_config.get('hidden_dims', [128, 64, 32])
            )
        else:
            self.model = AdaptiveTauNetwork(
                input_dim=model_config.get('input_dim', 20),
                hidden_dims=model_config.get('hidden_dims', [128, 64, 32]),
                dropout_rate=model_config.get('dropout', 0.2),
                use_batch_norm=model_config.get('batch_norm', True),
                use_residual=model_config.get('residual', True)
            )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params} total, {trainable_params} trainable")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        opt_config = self.config.get('optimizer', {})
        
        # Optimizer
        lr = opt_config.get('lr', 1e-3)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        if opt_config.get('type', 'adam') == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_config['type'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")
        
        # Learning rate scheduler
        if opt_config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=opt_config.get('lr_factor', 0.5),
                patience=opt_config.get('lr_patience', 10),
                min_lr=opt_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
    
    def _init_loss(self):
        """Initialize loss function"""
        loss_config = self.config.get('loss', {})
        
        self.criterion = AdaptiveTauLoss(
            safety_weight=loss_config.get('safety_weight', 10.0),
            efficiency_weight=loss_config.get('efficiency_weight', 1.0),
            smoothness_weight=loss_config.get('smoothness_weight', 0.5),
            coverage_weight=loss_config.get('coverage_weight', 2.0),
            violation_penalty=loss_config.get('violation_penalty', 5.0)
        )
        
        # Additional losses
        self.quantile_loss = QuantileLoss(quantile=0.9)
        self.mse_loss = WeightedMSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Training metrics
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            'safety_loss': [],
            'efficiency_loss': [],
            'coverage': [],
            'avg_tau': []
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move to device
            features = batch['features'].to(self.device)
            clearances = batch['clearance'].to(self.device)
            path_ids = batch.get('path_id', None)
            
            # Forward pass
            predicted_tau = self.model(features).squeeze()
            
            # Compute loss
            loss, loss_dict = self.criterion(
                predicted_tau, clearances.squeeze(),
                features, path_ids
            )
            
            # Add quantile loss
            quantile_loss = self.quantile_loss(predicted_tau, clearances.squeeze())
            loss += 0.5 * quantile_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            for key in ['safety_loss', 'efficiency_loss']:
                if key in loss_dict:
                    epoch_metrics[key].append(loss_dict[key])
            epoch_metrics['coverage'].append(loss_dict.get('empirical_coverage', 0))
            epoch_metrics['avg_tau'].append(loss_dict.get('avg_predicted_tau', 0))
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tau': f"{loss_dict.get('avg_predicted_tau', 0):.3f}"
            })
        
        # Compute epoch statistics
        metrics = {
            'loss': np.mean(epoch_losses),
            'safety_loss': np.mean(epoch_metrics['safety_loss']),
            'efficiency_loss': np.mean(epoch_metrics['efficiency_loss']),
            'coverage': np.mean(epoch_metrics['coverage']),
            'avg_tau': np.mean(epoch_metrics['avg_tau'])
        }
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """
        Validate model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = {
            'coverage': [],
            'avg_tau': [],
            'violations': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                features = batch['features'].to(self.device)
                clearances = batch['clearance'].to(self.device)
                
                # Forward pass
                predicted_tau = self.model(features).squeeze()
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    predicted_tau, clearances.squeeze(),
                    features
                )
                
                val_losses.append(loss.item())
                val_metrics['coverage'].append(loss_dict.get('empirical_coverage', 0))
                val_metrics['avg_tau'].append(loss_dict.get('avg_predicted_tau', 0))
                
                # Track violations
                violations = (predicted_tau < clearances.squeeze()).float().mean()
                val_metrics['violations'].append(violations.item())
        
        metrics = {
            'loss': np.mean(val_losses),
            'coverage': np.mean(val_metrics['coverage']),
            'avg_tau': np.mean(val_metrics['avg_tau']),
            'violation_rate': np.mean(val_metrics['violations'])
        }
        
        return metrics
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        logging.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['avg_tau'].append(train_metrics['avg_tau'])
            
            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['coverage'].append(val_metrics['coverage'])
                
                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_metrics['loss'])
                
                # Save best model
                if val_metrics['loss'] < self.best_loss:
                    self.best_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                
                logging.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Coverage={val_metrics['coverage']:.3f}, "
                    f"Avg Tau={train_metrics['avg_tau']:.3f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Avg Tau={train_metrics['avg_tau']:.3f}"
                )
            
            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        self.save_training_history()
        
        logging.info("Training completed")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Checkpoint loaded from {path}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logging.info(f"Training history saved to {history_path}")


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        (train_loader, val_loader)
    """
    # Initialize feature extractor
    feature_extractor = WaypointFeatureExtractor()
    
    # Load datasets
    train_dataset = PathPlanningDataset(
        config['data']['train_path'],
        feature_extractor
    )
    
    val_dataset = PathPlanningDataset(
        config['data']['val_path'],
        feature_extractor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    return train_loader, val_loader


def main():
    """Main training script"""
    # Configuration
    config = {
        'model': {
            'input_dim': 20,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2,
            'batch_norm': True,
            'residual': True,
            'use_ensemble': False
        },
        'optimizer': {
            'type': 'adam',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'use_scheduler': True,
            'lr_factor': 0.5,
            'lr_patience': 10
        },
        'loss': {
            'safety_weight': 10.0,
            'efficiency_weight': 1.0,
            'smoothness_weight': 0.5,
            'coverage_weight': 2.0,
            'violation_penalty': 5.0
        },
        'training': {
            'batch_size': 64,
            'num_epochs': 100,
            'num_workers': 4
        },
        'data': {
            'train_path': 'data/train_data.pkl',
            'val_path': 'data/val_data.pkl'
        },
        'checkpoint_dir': 'checkpoints'
    }
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create trainer
    trainer = LearnableCPTrainer(config)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Train model
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs']
    )


if __name__ == "__main__":
    main()